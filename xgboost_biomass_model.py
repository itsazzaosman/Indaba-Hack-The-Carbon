import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

class BiomassDataset(Dataset):
    """Dataset for biomass prediction from satellite imagery."""
    
    def __init__(self, csv_file, root_dir, transform=None, is_training=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the satellite images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_training (bool): Whether this is training data (affects augmentation)
        """
        self.biomass_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        
        # Extract sample IDs and biomass values
        self.sample_ids = self.biomass_data['ID'].values
        if 'biomass' in self.biomass_data.columns:
            self.biomass_values = self.biomass_data['biomass'].values
        else:
            # For test data, we might not have biomass values
            self.biomass_values = None
            
    def __len__(self):
        return len(self.biomass_data)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load satellite image (18 bands)
        img_path = os.path.join(self.root_dir, f"{sample_id}.tif")
        
        try:
            with rasterio.open(img_path) as src:
                # Read all bands (should be 18 bands)
                img = src.read()  # Shape: (bands, height, width)
                
                # Ensure we have 18 bands
                if img.shape[0] != 18:
                    # Pad or truncate to 18 bands if needed
                    if img.shape[0] < 18:
                        # Pad with zeros
                        padded = np.zeros((18, img.shape[1], img.shape[2]))
                        padded[:img.shape[0]] = img
                        img = padded
                    else:
                        # Truncate to first 18 bands
                        img = img[:18]
                
                # Convert to float32 and normalize
                img = img.astype(np.float32)
                
                # Apply normalization (using values from the config)
                mean_values = [251.35568237304688, 447.9508972167969, 452.6473388671875, 
                              1995.358154296875, 1629.1016845703125, 902.5798950195312,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Extended for 18 bands
                std_values = [144.5541229248047, 121.24767303466797, 188.44677734375,
                             505.7544860839844, 462.39678955078125, 367.0113525390625,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # Extended for 18 bands
                
                # Normalize each band
                for i in range(min(18, img.shape[0])):
                    if std_values[i] > 0:
                        img[i] = (img[i] - mean_values[i]) / std_values[i]
                
                # Convert to tensor
                img = torch.from_numpy(img)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a zero tensor as fallback
            img = torch.zeros((18, 256, 256), dtype=torch.float32)
        
        # Apply transformations if any
        if self.transform:
            img = self.transform(img)
        
        # Return image and biomass value (if available)
        if self.biomass_values is not None:
            biomass = torch.tensor(self.biomass_values[idx], dtype=torch.float32)
            return img, biomass
        else:
            return img, torch.tensor(0.0, dtype=torch.float32)  # Dummy value for test data

class CNNFeatureExtractor(nn.Module):
    """CNN for extracting features from satellite imagery."""
    
    def __init__(self, num_bands=18, feature_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        
        self.num_bands = num_bands
        self.feature_dim = feature_dim
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(num_bands, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection layer
        self.feature_proj = nn.Linear(512, feature_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Input shape: (batch_size, num_bands, height, width)
        
        # Convolutional feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 128x128
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 64x64
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 32x32
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)  # 16x16
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch_size, 512)
        
        # Feature projection
        x = self.feature_proj(x)
        x = self.dropout(x)
        
        return x

class XGBoostBiomassModel:
    """Combined CNN + XGBoost model for biomass prediction."""
    
    def __init__(self, feature_dim=512, learning_rate=0.1, max_depth=6, n_estimators=100):
        self.feature_dim = feature_dim
        self.cnn = CNNFeatureExtractor(feature_dim=feature_dim).to(device)
        self.xgb_model = xgb.XGBRegressor(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            gpu_id=0 if torch.cuda.is_available() else None,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, dataloader):
        """Extract features using the CNN."""
        self.cnn.eval()
        features = []
        biomass_values = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                if len(batch) == 2:
                    images, biomass = batch
                    images = images.to(device)
                    batch_features = self.cnn(images)
                    features.append(batch_features.cpu().numpy())
                    biomass_values.append(biomass.numpy())
                else:
                    images = batch[0].to(device)
                    batch_features = self.cnn(images)
                    features.append(batch_features.cpu().numpy())
        
        features = np.vstack(features)
        if biomass_values:
            biomass_values = np.concatenate(biomass_values)
            return features, biomass_values
        else:
            return features, None
    
    def train(self, train_dataloader, val_dataloader=None):
        """Train the model."""
        print("Extracting training features...")
        train_features, train_biomass = self.extract_features(train_dataloader)
        
        # Scale features
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        print("Training XGBoost model...")
        self.xgb_model.fit(train_features_scaled, train_biomass)
        
        # Validate if validation data is provided
        if val_dataloader:
            print("Extracting validation features...")
            val_features, val_biomass = self.extract_features(val_dataloader)
            val_features_scaled = self.scaler.transform(val_features)
            
            val_pred = self.xgb_model.predict(val_features_scaled)
            val_rmse = np.sqrt(np.mean((val_pred - val_biomass) ** 2))
            print(f"Validation RMSE: {val_rmse:.4f}")
        
        self.is_trained = True
        print("Training completed!")
    
    def predict(self, dataloader):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        print("Extracting features for prediction...")
        features, _ = self.extract_features(dataloader)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.xgb_model.predict(features_scaled)
        return predictions
    
    def save_model(self, path):
        """Save the trained model."""
        import joblib
        model_data = {
            'cnn_state_dict': self.cnn.state_dict(),
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'feature_dim': self.feature_dim
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model."""
        import joblib
        model_data = joblib.load(path)
        self.cnn.load_state_dict(model_data['cnn_state_dict'])
        self.xgb_model = model_data['xgb_model']
        self.scaler = model_data['scaler']
        self.feature_dim = model_data['feature_dim']
        self.is_trained = True
        print(f"Model loaded from {path}")

def create_dataloaders(train_csv, val_csv, root_dir, batch_size=32, num_workers=4):
    """Create training and validation dataloaders."""
    
    # Create datasets
    train_dataset = BiomassDataset(train_csv, root_dir, is_training=True)
    val_dataset = BiomassDataset(val_csv, root_dir, is_training=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

def main():
    """Main training function."""
    
    # Configuration
    config = {
        'train_csv': 'train.csv',  # Update with your actual path
        'val_csv': 'val.csv',      # Update with your actual path
        'root_dir': '.',           # Update with your actual path
        'batch_size': 32,
        'feature_dim': 512,
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 200,
        'num_workers': 4
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check if data files exist
    if not os.path.exists(config['train_csv']):
        print(f"Training CSV not found: {config['train_csv']}")
        print("Please update the configuration with correct file paths")
        return
    
    if not os.path.exists(config['val_csv']):
        print(f"Validation CSV not found: {config['val_csv']}")
        print("Please update the configuration with correct file paths")
        return
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config['train_csv'],
        config['val_csv'], 
        config['root_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create and train model
    print("Initializing model...")
    model = XGBoostBiomassModel(
        feature_dim=config['feature_dim'],
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth'],
        n_estimators=config['n_estimators']
    )
    
    # Train the model
    model.train(train_loader, val_loader)
    
    # Save the model
    model.save_model('xgboost_biomass_model.pkl')
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
