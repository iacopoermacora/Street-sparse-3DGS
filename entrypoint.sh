#!/bin/bash

set -e

# Set the TORCH_CUDA_ARCH_LIST environment variable
export TORCH_CUDA_ARCH_LIST="8.0"

# Add OpenCTM library to the library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# Install requirements
pip install -r /host/requirements.txt
echo "Installing requirements for mast3r and dust3r"
pip install -r /host/mast3r/requirements.txt
pip install -r /host/mast3r/dust3r/requirements.txt

# Install any additional Python packages if they weren't already installed in the Dockerfile
pip install cython # Required for mast3r_sfm
pip install laspy # Required for colmap conversion
pip install lazrs # Required for colmap conversion
pip install open3d trimesh # Additional packages requested
pip install pybind11 # Required for building the CTM exporter

echo "Requirements installed"

# Create setup.py in the ss_utils directory for building the CTM exporter
if [ -f "/host/ss_utils/ctm_exporter.cpp" ] && [ ! -f "/host/ss_utils/setup.py" ]; then
    echo "Creating setup.py for ctm_exporter in ss_utils folder"
    cat > /host/ss_utils/setup.py << 'EOL'
from setuptools import setup, Extension
import pybind11

ctm_exporter = Extension(
    "ctm_exporter",
    sources=["ctm_exporter.cpp"],
    include_dirs=[pybind11.get_include()],
    libraries=["openctm"],
    extra_compile_args=["-std=c++11"]
)

setup(
    name="ctm_exporter",
    version="0.1",
    description="CTM exporter module",
    ext_modules=[ctm_exporter]
)
EOL
fi

# Build and install the CTM exporter module
echo "Building CTM exporter module"
cd /host/ss_utils && python setup.py build_ext --inplace && python setup.py install --user

cd /host

echo "Container is running!!"

exec "$@"