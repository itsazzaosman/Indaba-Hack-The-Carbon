#!/usr/bin/env python3
"""
Single GPU training script for XGBoost biomass model
Use this to test basic functionality before running distributed training
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import rasterio
from tqdm import tqdm
import warnings
import time
from typing import Dict, Any, List, Tuple
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xgboost_biomass_model_advanced import (
    AdvancedCNNFeatureExtractor, 
    AdvancedBiomassDataset,
    AdvancedBiomassModel
)

def setup_device():
    """Setup device for training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device

def create_dataloaders(train_csv, val_csv, root_dir, batch_size, num_workers):
    """Create training and validation dataloaders."""
    print("Creating dataloaders...")
    
    # Training dataset with augmentation
    train_dataset = AdvancedBiomassDataset(
        train_csv, root_dir, is_training=True, augment_prob=0.5
    )
    
    # Validation dataset without augmentation
    val_dataset = AdvancedBiomassDataset(
        val_csv, root_dir, is_training=False, augment_prob=0.0
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions and targets
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
    
    # Calculate metrics
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()
    
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    mape = np.mean(np.abs((all_targets - all_predictions) / all_targets)) * 100
    
    metrics = {
        'val_loss': total_loss / num_batches,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    return metrics, all_predictions, all_targets

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Single GPU XGBoost Biomass Training')
    parser.add_argument('--train-csv', type=str, default='train.csv', help='Path to training CSV')
    parser.add_argument('--val-csv', type=str, default='val.csv', help='Path to validation CSV')
    parser.add_argument('--root-dir', type=str, default='.', help='Root directory for data')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--feature-dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Print current working directory and script location for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Arguments: {args}")
    
    # Setup device
    device = setup_device()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.train_csv, args.val_csv, args.root_dir, 
        args.batch_size, args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = AdvancedBiomassModel(
        feature_dim=args.feature_dim,
        num_bands=18
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_metrics, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_metrics['val_loss'])
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val MSE: {val_metrics['mse']:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}")
        print(f"Val R²: {val_metrics['r2']:.4f}")
        print(f"Val MAPE: {val_metrics['mape']:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print("✓ New best model saved!")
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_time
    }
    
    np.save(os.path.join(args.save_dir, 'training_history.npy'), history)
    print("Training history saved")

if __name__ == "__main__":
    main()
