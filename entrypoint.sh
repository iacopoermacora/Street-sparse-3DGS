#!/bin/bash

set -e

pip install -r /host/requirements.txt
echo "Installing requirements for mast3r and dust3r"
pip install -r /host/mast3r/requirements.txt
pip install -r /host/mast3r/dust3r/requirements.txt
echo "Requirements installed"

echo "Container is running!!"

exec "$@"