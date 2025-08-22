#!/usr/bin/env python3
"""
8-GPU Training Script for XGBoost Biomass Model
Optimized for large-scale satellite imagery processing
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import rasterio
from tqdm import tqdm
import warnings
import time
from typing import Dict, Any, List, Tuple
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xgboost_biomass_model_advanced import (
    AdvancedBiomassDataset, 
    AdvancedCNNFeatureExtractor,
    ResNetBlock
)

def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize the process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        print(f"Distributed training: rank {rank}/{world_size}, local rank {local_rank}")
        return device, rank, world_size, local_rank
    else:
        print("Single GPU training")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 1, 0

class DistributedBiomassModel(nn.Module):
    """Distributed biomass prediction model for 8-GPU training."""
    
    def __init__(self, feature_dim=1024, num_bands=18):
        super(DistributedBiomassModel, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_bands = num_bands
        
        # CNN feature extractor
        self.cnn = AdvancedCNNFeatureExtractor(
            num_bands=num_bands, 
            feature_dim=feature_dim,
            use_attention=True
        )
        
        # Feature projection to final output
        self.final_proj = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # Extract features
        features = self.cnn(x)
        
        # Final prediction
        output = self.final_proj(features)
        
        return output.squeeze(-1)

def create_distributed_dataloaders(
    train_csv: str, 
    val_csv: str, 
    root_dir: str, 
    batch_size: int,
    world_size: int,
    rank: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create distributed dataloaders for multi-GPU training."""
    
    # Create datasets
    train_dataset = AdvancedBiomassDataset(
        train_csv, root_dir, is_training=True, augment_prob=0.7, noise_factor=0.05
    )
    val_dataset = AdvancedBiomassDataset(
        val_csv, root_dir, is_training=False, augment_prob=0.0, noise_factor=0.0
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )
    
    return train_loader, val_loader, train_sampler

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    rank: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Set sampler epoch for proper shuffling
    train_loader.sampler.set_epoch(epoch)
    
    progress_bar = tqdm(
        train_loader, 
        desc=f'Epoch {epoch} (Rank {rank})',
        disable=rank != 0
    )
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0:
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rank: int
) -> Tuple[float, List[float], List[float]]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss, all_predictions, all_targets

def calculate_metrics(predictions: List[float], targets: List[float]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / np.where(targets != 0, targets, 1))) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def main():
    # Print current working directory and script location for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    
    parser = argparse.ArgumentParser(description='8-GPU XGBoost Biomass Training')
    parser.add_argument('--train-csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--val-csv', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--root-dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--feature-dim', type=int, default=1024, help='Feature dimension')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers per GPU')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Checkpoint save directory')
    
    args = parser.parse_args()
    
    # Print arguments for debugging
    print(f"Arguments: {args}")
    
    # Setup distributed training
    device, rank, world_size, local_rank = setup_distributed()
    
    # Create save directory
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Create dataloaders
    print(f"Rank {rank}: Creating dataloaders...")
    train_loader, val_loader, train_sampler = create_distributed_dataloaders(
        args.train_csv,
        args.val_csv,
        args.root_dir,
        args.batch_size,
        world_size,
        rank,
        args.num_workers
    )
    
    # Create model
    print(f"Rank {rank}: Creating model...")
    model = DistributedBiomassModel(
        feature_dim=args.feature_dim,
        num_bands=18
    )
    
    # Move model to device
    model = model.to(device)
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
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
    
    print(f"Rank {rank}: Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, rank
        )
        
        # Validate
        val_loss, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device, rank
        )
        
        # Update learning rate
        scheduler.step()
        
        # Gather metrics from all processes
        if world_size > 1:
            # Gather validation loss
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / world_size
            
            # Gather predictions and targets for metrics calculation
            # This is simplified - in practice you might want to gather all predictions
            if rank == 0:
                val_preds = np.array(val_preds)
                val_targets = np.array(val_targets)
        
        # Log metrics (only on rank 0)
        if rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{args.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
            
            # Calculate additional metrics on validation set
            if epoch % 5 == 0:  # Every 5 epochs
                metrics = calculate_metrics(val_preds, val_targets)
                print(f'  Val RMSE: {metrics["rmse"]:.4f}')
                print(f'  Val R²: {metrics["r2"]:.4f}')
                print(f'  Val MAE: {metrics["mae"]:.4f}')
        
        # Save best model
        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'  Saved best model with validation loss: {best_val_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if rank == 0 and (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Training completed
    total_time = time.time() - start_time
    if rank == 0:
        print(f'\nTraining completed in {total_time/3600:.2f} hours')
        print(f'Best validation loss: {best_val_loss:.4f}')
        
        # Save final model
        final_checkpoint = {
            'epoch': args.epochs,
            'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'args': args
        }
        torch.save(final_checkpoint, os.path.join(args.save_dir, 'final_model.pth'))
        print('Final model saved')
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
