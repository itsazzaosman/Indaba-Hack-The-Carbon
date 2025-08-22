# XGBoost + CNN Biomass Prediction Model

This repository contains an innovative approach to biomass prediction from satellite imagery that combines the power of Convolutional Neural Networks (CNNs) for feature extraction with XGBoost for final prediction. This hybrid approach leverages the best of both worlds: deep learning for spatial feature extraction and gradient boosting for robust regression.

## 🚀 Key Features

- **CNN Feature Extraction**: Advanced ResNet-style architecture with attention mechanisms
- **XGBoost Regression**: GPU-accelerated gradient boosting for final predictions
- **Multi-GPU Support**: Distributed training across 8 GPUs using PyTorch DDP
- **Data Augmentation**: Comprehensive augmentation strategies for satellite imagery
- **Ensemble Methods**: Combines XGBoost with Random Forest for improved performance
- **Automatic Normalization**: Calculates normalization statistics from your data

## 🏗️ Architecture Overview

### 1. CNN Feature Extractor
The CNN component processes 18-band satellite imagery (256x256 pixels) through:
- **ResNet-style blocks** with residual connections
- **Multi-head attention** mechanisms for spatial relationships
- **Global average pooling** to extract compact features
- **Feature projection** layers to reduce dimensionality

### 2. XGBoost Regressor
The XGBoost component:
- Takes CNN-extracted features as input
- Uses GPU acceleration (`tree_method='gpu_hist'`)
- Implements advanced regularization techniques
- Provides feature importance analysis

### 3. Ensemble Approach
Combines multiple models:
- **XGBoost** (60% weight)
- **Random Forest** (40% weight)
- **Robust scaling** for feature normalization

## 📊 Data Requirements

### Input Format
- **Satellite Images**: 18-band GeoTIFF files (256x256 pixels)
- **Training CSV**: Columns: `ID`, `biomass`
- **Validation CSV**: Same format as training
- **Image Naming**: `{ID}.tif` format

### Data Structure
```
project_root/
├── train.csv          # Training data with ID and biomass columns
├── val.csv            # Validation data
├── images/            # Directory containing .tif files
│   ├── sample1.tif
│   ├── sample2.tif
│   └── ...
└── checkpoints/       # Model checkpoints (created automatically)
```

## 🛠️ Installation

### 1. Install Dependencies
```bash
pip install -r requirements_xgboost.txt
```

### 2. Verify GPU Setup
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 3. Install XGBoost with GPU Support
```bash
pip install xgboost --upgrade
```

## 🚀 Usage

### Single GPU Training
```bash
python xgboost_biomass_model_advanced.py
```

### 8-GPU Distributed Training
```bash
# Make launch script executable
chmod +x launch_8gpu_training.sh

# Launch training
./launch_8gpu_training.sh
```

### Manual Distributed Training
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    train_8gpu.py \
    --train-csv train.csv \
    --val-csv val.csv \
    --root-dir ./images \
    --batch-size 8 \
    --epochs 50
```

## ⚙️ Configuration

### Training Parameters
- **Batch Size**: 8 per GPU (64 total for 8 GPUs)
- **Learning Rate**: 1e-4 (with cosine annealing)
- **Feature Dimension**: 1024
- **Epochs**: 50
- **Weight Decay**: 1e-4

### Model Architecture
- **CNN Layers**: 4 ResNet blocks with attention
- **Feature Extraction**: 1024-dimensional features
- **Dropout**: 0.3 for regularization
- **Optimizer**: AdamW with weight decay

## 📈 Performance Features

### 1. GPU Acceleration
- **PyTorch**: CNN training on multiple GPUs
- **XGBoost**: GPU-accelerated tree building
- **Data Loading**: Pinned memory and persistent workers

### 2. Memory Optimization
- **Gradient Checkpointing**: Reduces memory usage
- **Mixed Precision**: Optional FP16 training
- **Efficient Attention**: Memory-efficient attention mechanisms

### 3. Scalability
- **Distributed Training**: Scales across multiple GPUs
- **Data Parallelism**: Efficient data distribution
- **Model Parallelism**: Optional for very large models

## 🔍 Model Analysis

### Feature Importance
The model provides insights into:
- **Band Importance**: Which satellite bands are most predictive
- **Spatial Features**: What spatial patterns matter most
- **Model Confidence**: Uncertainty estimates for predictions

### Visualization
- **Prediction vs Actual**: Scatter plots for model evaluation
- **Training Curves**: Loss and metric tracking
- **Feature Maps**: CNN attention visualization

## 📊 Expected Performance

### Training Time
- **Single GPU**: ~2-4 hours for 50 epochs
- **8 GPUs**: ~15-30 minutes for 50 epochs
- **Speedup**: 8-16x faster with distributed training

### Memory Usage
- **Per GPU**: 8-12 GB VRAM
- **Total**: 64-96 GB VRAM across 8 GPUs
- **Batch Size**: 64 total (8 per GPU)

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size per GPU
   - Enable gradient checkpointing
   - Use mixed precision training

2. **NCCL Communication Errors**
   - Check network configuration
   - Disable P2P if needed
   - Verify GPU topology

3. **Data Loading Bottlenecks**
   - Increase number of workers
   - Use SSD storage for images
   - Enable persistent workers

### Performance Tips

1. **Data Preprocessing**
   - Pre-calculate normalization statistics
   - Use memory-mapped files for large datasets
   - Implement data caching

2. **Training Optimization**
   - Use learning rate scheduling
   - Implement early stopping
   - Monitor validation metrics

3. **Hardware Optimization**
   - Ensure sufficient CPU cores for data loading
   - Use high-speed storage for images
   - Monitor GPU utilization

## 🔬 Research Applications

This approach is particularly suitable for:
- **Large-scale biomass mapping**
- **Multi-temporal analysis**
- **Transfer learning to new regions**
- **Ensemble model development**
- **Feature importance analysis**

## 📚 References

- **XGBoost**: Chen & Guestrin (2016) - Gradient Boosting with XGBoost
- **ResNet**: He et al. (2016) - Deep Residual Learning
- **Attention Mechanisms**: Vaswani et al. (2017) - Attention is All You Need
- **Distributed Training**: Li et al. (2020) - PyTorch Distributed

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **InstaGeo Team** for the original framework
- **PyTorch Team** for distributed training capabilities
- **XGBoost Team** for GPU acceleration support
- **Open Source Community** for various dependencies

---

**Note**: This approach represents a significant departure from traditional Vision Transformer methods, offering potentially better interpretability and faster training while maintaining competitive performance. The CNN + XGBoost combination is particularly well-suited for regression tasks where spatial feature extraction is crucial.
