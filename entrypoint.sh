#!/bin/bash

set -e

# Set the TORCH_CUDA_ARCH_LIST environment variable
export TORCH_CUDA_ARCH_LIST="8.0"

pip install -r /host/requirements.txt
echo "Installing requirements for mast3r and dust3r"
pip install -r /host/mast3r/requirements.txt
pip install -r /host/mast3r/dust3r/requirements.txt
pip install cython # Required for mast3r_sfm
pip install laspy # Required for colmap conversion
pip install lazrs # Required for colmap conversion
echo "Requirements installed"

echo "Container is running!!"

exec "$@"