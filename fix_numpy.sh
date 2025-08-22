#!/bin/bash

# Fix NumPy compatibility issue
echo "Fixing NumPy compatibility issue..."

# Check current NumPy version
echo "Current NumPy version:"
python3 -c "import numpy; print(numpy.__version__)"

# Downgrade NumPy to compatible version
echo "Downgrading NumPy to compatible version..."
pip install "numpy<2.0.0" --force-reinstall

# Verify the fix
echo "Verifying NumPy version:"
python3 -c "import numpy; print(numpy.__version__)"

echo "NumPy compatibility issue fixed!"
echo "You can now run the training scripts."
