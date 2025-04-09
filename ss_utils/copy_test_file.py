'''
Thesis Project: Street-sparse-3DGS
Author: Iacopo Ermacora
Date: 11/2024-06/2025

Description: This script copies the test.txt file from the ss_raw_images directory to the 
aligned and chunks directories. The test.txt file contains the list of images to be processed as
test images.
'''

import os
import shutil
import glob
import argparse
import sys

if __name__ == "__main__":
    # Add the parent directory to the system path
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--project_dir", required=True, type=str, help="Path to the project directory")
    args = argparse.parse_args()

    # Define all the paths
    test_txt_path = os.path.join(args.project_dir, "ss_raw_images", "test.txt")
    aligned_path = os.path.join(args.project_dir, "camera_calibration", "aligned", "sparse", "0")
    chunks_path = os.path.join(args.project_dir, "camera_calibration", "chunks")

    # Check if the test.txt file exists
    if not os.path.exists(test_txt_path):
        print(f"Error: {test_txt_path} does not exist.")
        sys.exit(1)
    
    # Copy the test.txt file to the aligned path
    shutil.copy(test_txt_path, aligned_path)
    print(f"Copied {test_txt_path} to {aligned_path}")

    # Copy the test.txt file to each chunk in the chunks path
    chunks_paths = os.listdir(chunks_path)
    for chunk in chunks_paths:
        chunk_path = os.path.join(chunks_path, chunk, "sparse", "0")
        if os.path.exists(chunk_path):
            shutil.copy(test_txt_path, chunk_path)
            print(f"Copied {test_txt_path} to {chunk_path}")
        else:
            print(f"Warning: {chunk_path} does not exist. Skipping.")