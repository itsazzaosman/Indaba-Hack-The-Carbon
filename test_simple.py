#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import os
import sys
import torch
import numpy as np

def test_basic_setup():
    """Test basic PyTorch and GPU setup."""
    print("=== Basic Setup Test ===")
    
    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Test basic tensor operations
    print("\nTesting basic tensor operations...")
    x = torch.randn(2, 3).cuda() if torch.cuda.is_available() else torch.randn(2, 3)
    y = torch.randn(2, 3).cuda() if torch.cuda.is_available() else torch.randn(2, 3)
    z = x + y
    print(f"Tensor addition successful: {z.shape}")
    
    return True

def test_model_import():
    """Test if we can import the model components."""
    print("\n=== Model Import Test ===")
    
    try:
        from xgboost_biomass_model_advanced import AdvancedCNNFeatureExtractor
        print("✓ CNN feature extractor imported successfully")
        
        # Test model creation
        cnn = AdvancedCNNFeatureExtractor(num_bands=18, feature_dim=512)
        print("✓ CNN model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(2, 18, 256, 256)
        with torch.no_grad():
            features = cnn(dummy_input)
            print(f"✓ CNN forward pass successful, output shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\n=== Data Loading Test ===")
    
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
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting simple functionality tests...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    
    tests = [
        ("Basic Setup", test_basic_setup),
        ("Model Import", test_model_import),
        ("Data Loading", test_data_loading)
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
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Basic functionality is working.")
        print("\nYou can now try the distributed training:")
        print("1. Make sure you have train.csv and val.csv files")
        print("2. Run: sudo bash ./launch_8gpu_training.sh")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
