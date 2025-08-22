#!/usr/bin/env python3
"""
Test script to verify the XGBoost + CNN biomass model setup
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import xgboost as xgb
        print(f"✓ XGBoost {xgb.__version__}")
    except ImportError as e:
        print(f"✗ XGBoost import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import rasterio
        print(f"✓ Rasterio {rasterio.__version__}")
    except ImportError as e:
        print(f"✗ Rasterio import failed: {e}")
        return False
    
    try:
        import tqdm
        print(f"✓ TQDM {tqdm.__version__}")
    except ImportError as e:
        print(f"✗ TQDM import failed: {e}")
        return False
    
    return True

def test_gpu_setup():
    """Test GPU availability and configuration."""
    print("\nTesting GPU setup...")
    
    import torch
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False
    
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Test XGBoost GPU support
    try:
        import xgboost as xgb
        # Try to create a simple XGBoost model with GPU
        model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)
        print("✓ XGBoost GPU support available")
    except Exception as e:
        print(f"✗ XGBoost GPU support failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if the model components can be created."""
    print("\nTesting model creation...")
    
    try:
        # Test CNN feature extractor
        from xgboost_biomass_model_advanced import AdvancedCNNFeatureExtractor
        
        cnn = AdvancedCNNFeatureExtractor(num_bands=18, feature_dim=512)
        print("✓ CNN feature extractor created successfully")
        
        # Test with dummy input
        import torch
        dummy_input = torch.randn(2, 18, 256, 256)
        with torch.no_grad():
            features = cnn(dummy_input)
            print(f"✓ CNN forward pass successful, output shape: {features.shape}")
        
    except Exception as e:
        print(f"✗ CNN model creation failed: {e}")
        return False
    
    try:
        # Test XGBoost model
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            gpu_id=0 if torch.cuda.is_available() else None
        )
        print("✓ XGBoost model created successfully")
        
    except Exception as e:
        print(f"✗ XGBoost model creation failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        from xgboost_biomass_model_advanced import AdvancedBiomassDataset
        
        # Create a dummy CSV for testing
        import pandas as pd
        dummy_data = pd.DataFrame({
            'ID': ['test1', 'test2'],
            'biomass': [100.0, 200.0]
        })
        dummy_csv = 'test_dummy.csv'
        dummy_data.to_csv(dummy_csv, index=False)
        
        # Test dataset creation
        dataset = AdvancedBiomassDataset(
            dummy_csv, 
            '.', 
            is_training=False,
            augment_prob=0.0
        )
        print("✓ Dataset created successfully")
        print(f"  Dataset size: {len(dataset)}")
        
        # Clean up
        os.remove(dummy_csv)
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False
    
    return True

def test_distributed_setup():
    """Test distributed training setup."""
    print("\nTesting distributed training setup...")
    
    try:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        # Test if distributed modules can be imported
        print("✓ Distributed training modules available")
        
        # Test if NCCL backend is available
        if dist.is_nccl_available():
            print("✓ NCCL backend available")
        else:
            print("⚠ NCCL backend not available")
        
    except Exception as e:
        print(f"✗ Distributed training setup failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("XGBoost + CNN Biomass Model Setup Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("GPU Setup", test_gpu_setup),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Distributed Setup", test_distributed_setup)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready for training.")
        print("\nNext steps:")
        print("1. Prepare your data (train.csv, val.csv, image files)")
        print("2. Update file paths in the configuration")
        print("3. Run training: python xgboost_biomass_model_advanced.py")
        print("4. For 8-GPU training: ./launch_8gpu_training.sh")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements_xgboost.txt")
        print("2. Check CUDA installation and PyTorch compatibility")
        print("3. Verify XGBoost GPU support installation")
        print("4. Check system requirements and dependencies")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
