import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
warnings.filterwarnings('ignore')

# Set device for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    # Enable memory efficient attention if available
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class AdvancedBiomassDataset(Dataset):
    """Advanced dataset for biomass prediction with data augmentation."""
    
    def __init__(self, csv_file, root_dir, transform=None, is_training=True, 
                 augment_prob=0.5, noise_factor=0.1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the satellite images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_training (bool): Whether this is training data (affects augmentation)
            augment_prob (float): Probability of applying augmentation
            noise_factor (float): Factor for noise injection
        """
        self.biomass_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        self.augment_prob = augment_prob
        self.noise_factor = noise_factor
        
        # Extract sample IDs and biomass values
        self.sample_ids = self.biomass_data['ID'].values
        if 'biomass' in self.biomass_data.columns:
            self.biomass_values = self.biomass_data['biomass'].values
        else:
            self.biomass_values = None
            
        # Calculate statistics for normalization
        self._calculate_normalization_stats()
            
    def _calculate_normalization_stats(self):
        """Calculate normalization statistics from a sample of images."""
        print("Calculating normalization statistics...")
        sample_size = min(100, len(self.sample_ids))
        sample_indices = np.random.choice(len(self.sample_ids), sample_size, replace=False)
        
        all_means = []
        all_stds = []
        
        for idx in tqdm(sample_indices, desc="Calculating stats"):
            sample_id = self.sample_ids[idx]
            img_path = os.path.join(self.root_dir, f"{sample_id}.tif")
            
            try:
                with rasterio.open(img_path) as src:
                    img = src.read()
                    if img.shape[0] > 0:
                        all_means.append(np.mean(img, axis=(1, 2)))
                        all_stds.append(np.std(img, axis=(1, 2)))
            except:
                continue
        
        if all_means:
            self.mean_values = np.mean(all_means, axis=0)
            self.std_values = np.mean(all_stds, axis=0)
            # Ensure no division by zero
            self.std_values = np.where(self.std_values == 0, 1.0, self.std_values)
            print(f"Calculated mean: {self.mean_values[:6]}...")  # Show first 6 bands
            print(f"Calculated std: {self.std_values[:6]}...")
        else:
            # Fallback to default values
            self.mean_values = np.array([251.35568237304688, 447.9508972167969, 452.6473388671875, 
                                        1995.358154296875, 1629.1016845703125, 902.5798950195312,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            self.std_values = np.array([144.5541229248047, 121.24767303466797, 188.44677734375,
                                       505.7544860839844, 462.39678955078125, 367.0113525390625,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            
    def __len__(self):
        return len(self.biomass_data)
    
    def _augment_image(self, img):
        """Apply data augmentation to the image."""
        if not self.is_training or np.random.random() > self.augment_prob:
            return img
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.flip(img, axis=2)
        
        # Random vertical flip
        if np.random.random() > 0.5:
            img = np.flip(img, axis=1)
        
        # Random rotation (90, 180, 270 degrees)
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            img = np.rot90(img, k=k, axes=(1, 2))
        
        # Add random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, self.noise_factor, img.shape)
            img = img + noise
            img = np.clip(img, 0, 1)  # Ensure values stay in reasonable range
        
        return img
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load satellite image (18 bands)
        img_path = os.path.join(self.root_dir, f"{sample_id}.tif")
        
        try:
            with rasterio.open(img_path) as src:
                img = src.read()  # Shape: (bands, height, width)
                
                # Ensure we have 18 bands
                if img.shape[0] != 18:
                    if img.shape[0] < 18:
                        padded = np.zeros((18, img.shape[1], img.shape[2]))
                        padded[:img.shape[0]] = img
                        img = padded
                    else:
                        img = img[:18]
                
                # Convert to float32
                img = img.astype(np.float32)
                
                # Apply normalization using calculated statistics
                for i in range(min(18, img.shape[0])):
                    if self.std_values[i] > 0:
                        img[i] = (img[i] - self.mean_values[i]) / self.std_values[i]
                
                # Apply augmentation
                img = self._augment_image(img)
                
                # Convert to tensor
                img = torch.from_numpy(img)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = torch.zeros((18, 256, 256), dtype=torch.float32)
        
        # Apply transformations if any
        if self.transform:
            img = self.transform(img)
        
        # Return image and biomass value (if available)
        if self.biomass_values is not None:
            biomass = torch.tensor(self.biomass_values[idx], dtype=torch.float32)
            return img, biomass
        else:
            return img, torch.tensor(0.0, dtype=torch.float32)

class ResNetBlock(nn.Module):
    """ResNet-style residual block."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AdvancedCNNFeatureExtractor(nn.Module):
    """Advanced CNN with ResNet-style blocks for feature extraction."""
    
    def __init__(self, num_bands=18, feature_dim=1024, use_attention=True):
        super(AdvancedCNNFeatureExtractor, self).__init__()
        
        self.num_bands = num_bands
        self.feature_dim = feature_dim
        self.use_attention = use_attention
        
        # Initial convolution to reduce band dimension
        self.initial_conv = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3)
        self.initial_bn = nn.BatchNorm2d(64)
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-style blocks
        self.layer1 = self._make_layer(64, 128, 2, stride=1)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        self.layer4 = self._make_layer(512, 1024, 2, stride=2)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(1024, num_heads=8, batch_first=True)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection layers
        self.feature_proj = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input shape: (batch_size, num_bands, height, width)
        
        # Initial convolution and pooling
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        x = self.initial_pool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention: (batch_size, channels, height*width)
            batch_size, channels, height, width = x.size()
            x_reshaped = x.view(batch_size, channels, -1).permute(0, 2, 1)
            
            # Apply self-attention
            x_attended, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
            x = x_attended.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Feature projection
        x = self.feature_proj(x)
        
        return x

class EnsembleBiomassModel:
    """Ensemble model combining CNN features with multiple ML algorithms."""
    
    def __init__(self, feature_dim=1024, use_multi_gpu=True):
        self.feature_dim = feature_dim
        self.use_multi_gpu = use_multi_gpu
        
        # Initialize CNN
        self.cnn = AdvancedCNNFeatureExtractor(feature_dim=feature_dim)
        
        # Move to GPU and wrap with DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1 and use_multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.cnn = DataParallel(self.cnn)
        
        self.cnn = self.cnn.to(device)
        
        # Initialize multiple ML models
        self.xgb_model = xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=8,
            n_estimators=300,
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            gpu_id=0 if torch.cuda.is_available() else None,
            random_state=42,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Scaler for features
        self.scaler = RobustScaler()
        
        # Ensemble weights
        self.ensemble_weights = [0.6, 0.4]  # XGBoost, Random Forest
        
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
        """Train the ensemble model."""
        print("Extracting training features...")
        train_features, train_biomass = self.extract_features(train_dataloader)
        
        # Scale features
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        print("Training XGBoost model...")
        self.xgb_model.fit(train_features_scaled, train_biomass)
        
        print("Training Random Forest model...")
        self.rf_model.fit(train_features_scaled, train_biomass)
        
        # Validate if validation data is provided
        if val_dataloader:
            print("Extracting validation features...")
            val_features, val_biomass = self.extract_features(val_dataloader)
            val_features_scaled = self.scaler.transform(val_features)
            
            # Individual model predictions
            xgb_pred = self.xgb_model.predict(val_features_scaled)
            rf_pred = self.rf_model.predict(val_features_scaled)
            
            # Ensemble prediction
            ensemble_pred = (self.ensemble_weights[0] * xgb_pred + 
                           self.ensemble_weights[1] * rf_pred)
            
            # Calculate metrics
            xgb_rmse = np.sqrt(np.mean((xgb_pred - val_biomass) ** 2))
            rf_rmse = np.sqrt(np.mean((rf_pred - val_biomass) ** 2))
            ensemble_rmse = np.sqrt(np.mean((ensemble_pred - val_biomass) ** 2))
            
            print(f"Validation RMSE - XGBoost: {xgb_rmse:.4f}")
            print(f"Validation RMSE - Random Forest: {rf_rmse:.4f}")
            print(f"Validation RMSE - Ensemble: {ensemble_rmse:.4f}")
            
            # Plot predictions vs actual
            self._plot_predictions(val_biomass, xgb_pred, rf_pred, ensemble_pred)
        
        self.is_trained = True
        print("Training completed!")
    
    def predict(self, dataloader):
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        print("Extracting features for prediction...")
        features, _ = self.extract_features(dataloader)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Individual model predictions
        xgb_pred = self.xgb_model.predict(features_scaled)
        rf_pred = self.rf_model.predict(features_scaled)
        
        # Ensemble prediction
        ensemble_pred = (self.ensemble_weights[0] * xgb_pred + 
                        self.ensemble_weights[1] * rf_pred)
        
        return ensemble_pred, xgb_pred, rf_pred
    
    def _plot_predictions(self, actual, xgb_pred, rf_pred, ensemble_pred):
        """Plot predictions vs actual values."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XGBoost predictions
        axes[0].scatter(actual, xgb_pred, alpha=0.6)
        axes[0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Biomass')
        axes[0].set_ylabel('Predicted Biomass')
        axes[0].set_title('XGBoost Predictions')
        axes[0].grid(True, alpha=0.3)
        
        # Random Forest predictions
        axes[1].scatter(actual, rf_pred, alpha=0.6)
        axes[1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Biomass')
        axes[1].set_ylabel('Predicted Biomass')
        axes[1].set_title('Random Forest Predictions')
        axes[1].grid(True, alpha=0.3)
        
        # Ensemble predictions
        axes[2].scatter(actual, ensemble_pred, alpha=0.6)
        axes[2].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[2].set_xlabel('Actual Biomass')
        axes[2].set_ylabel('Predicted Biomass')
        axes[2].set_title('Ensemble Predictions')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path):
        """Save the trained model."""
        import joblib
        model_data = {
            'cnn_state_dict': self.cnn.state_dict(),
            'xgb_model': self.xgb_model,
            'rf_model': self.rf_model,
            'scaler': self.scaler,
            'feature_dim': self.feature_dim,
            'ensemble_weights': self.ensemble_weights
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model."""
        import joblib
        model_data = joblib.load(path)
        self.cnn.load_state_dict(model_data['cnn_state_dict'])
        self.xgb_model = model_data['xgb_model']
        self.rf_model = model_data['rf_model']
        self.scaler = model_data['scaler']
        self.feature_dim = model_data['feature_dim']
        self.ensemble_weights = model_data['ensemble_weights']
        self.is_trained = True
        print(f"Model loaded from {path}")

def create_advanced_dataloaders(train_csv, val_csv, root_dir, batch_size=16, num_workers=4):
    """Create training and validation dataloaders with advanced features."""
    
    # Create datasets
    train_dataset = AdvancedBiomassDataset(
        train_csv, root_dir, is_training=True, augment_prob=0.7, noise_factor=0.05
    )
    val_dataset = AdvancedBiomassDataset(
        val_csv, root_dir, is_training=False, augment_prob=0.0, noise_factor=0.0
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader

def main():
    """Main training function."""
    
    # Configuration
    config = {
        'train_csv': 'train.csv',  # Update with your actual path
        'val_csv': 'val.csv',      # Update with your actual path
        'root_dir': '.',           # Update with your actual path
        'batch_size': 16,          # Reduced for larger model
        'feature_dim': 1024,       # Increased feature dimension
        'num_workers': 4,
        'use_multi_gpu': True
    }
    
    print("Advanced XGBoost Biomass Model Configuration:")
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
    print("Creating advanced dataloaders...")
    train_loader, val_loader = create_advanced_dataloaders(
        config['train_csv'],
        config['val_csv'], 
        config['root_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create and train model
    print("Initializing advanced ensemble model...")
    model = EnsembleBiomassModel(
        feature_dim=config['feature_dim'],
        use_multi_gpu=config['use_multi_gpu']
    )
    
    # Train the model
    model.train(train_loader, val_loader)
    
    # Save the model
    model.save_model('advanced_ensemble_biomass_model.pkl')
    
    print("Advanced training completed successfully!")

if __name__ == "__main__":
    main()
