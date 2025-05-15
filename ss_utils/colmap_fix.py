#!/usr/bin/env python3
import os
import shutil
import sys
import argparse
from pathlib import Path
from read_write_model import *

def process_test_file(project_dir):
    """Process test.txt file - delete and rename camera entries"""
    test_file_path = os.path.join(project_dir, "test.txt")
    
    if not os.path.exists(test_file_path):
        print(f"Test file {test_file_path} does not exist")
        return
    
    print(f"Processing test file: {test_file_path}")
    
    # Read the test file
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
    
    # Process lines
    new_lines = []
    for line in lines:
        line = line.strip()
        
        # Skip lines containing cameras to delete
        if any(f"cam{cam}" in line for cam in [2, 4, 6, 8]):
            print(f"Deleting line: {line}")
            continue
        
        # Rename cameras in each line
        new_line = line
        for src, dst in [("cam3", "cam2"), ("cam5", "cam3"), ("cam7", "cam4"), ("cam9", "cam5"), ("cam10", "cam6")]:
            if src in new_line:
                new_line = new_line.replace(src, dst)
                print(f"Renaming in line: {line} -> {new_line}")
                break
        
        new_lines.append(new_line + '\n')
    
    # Write back to the file
    with open(test_file_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Test file updated with {len(new_lines)} lines")

def process_folders(base_path, folder_type):
    """Process image or mask folders - delete and rename as specified"""
    path = Path(base_path) / folder_type
    
    if not path.exists() or not path.is_dir():
        print(f"{path} does not exist or is not a directory")
        return
    
    # Folders to delete
    for folder_name in ["cam2", "cam4", "cam6", "cam8"]:
        folder_path = path / folder_name
        if folder_path.exists():
            print(f"Deleting {folder_path}")
            shutil.rmtree(folder_path)
    
    # Rename folders
    rename_map = {
        "cam3": "cam2",
        "cam5": "cam3",
        "cam7": "cam4",
        "cam9": "cam5",
        "cam10": "cam6"
    }
    
    # Create a list of temporary names first to avoid conflicts during renaming
    for src, dst in rename_map.items():
        src_path = path / src
        if src_path.exists():
            temp_path = path / f"temp_{src}"
            print(f"Renaming {src_path} to {temp_path} (temporary)")
            src_path.rename(temp_path)
    
    # Now rename from temporary names to final names
    for src, dst in rename_map.items():
        temp_path = path / f"temp_{src}"
        if temp_path.exists():
            dst_path = path / dst
            print(f"Renaming {temp_path} to {dst_path} (final)")
            temp_path.rename(dst_path)

def process_test_file(project_dir):
    """Process test.txt file - delete and rename camera entries"""
    test_file_path = os.path.join(project_dir, "test.txt")
    
    if not os.path.exists(test_file_path):
        print(f"Test file {test_file_path} does not exist")
        return
    
    print(f"Processing test file: {test_file_path}")
    
    # Read the test file
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
    
    # Process lines
    new_lines = []
    for line in lines:
        line = line.strip()
        
        # Skip lines containing cameras to delete
        if any(f"cam{cam}" in line for cam in [2, 4, 6, 8]):
            print(f"Deleting line: {line}")
            continue
        
        # Rename cameras in each line
        new_line = line
        for src, dst in [("cam3", "cam2"), ("cam5", "cam3"), ("cam7", "cam4"), ("cam9", "cam5"), ("cam10", "cam6")]:
            if src in new_line:
                new_line = new_line.replace(src, dst)
                print(f"Renaming in line: {line} -> {new_line}")
                break
        
        new_lines.append(new_line + '\n')
    
    # Write back to the file
    with open(test_file_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Test file updated with {len(new_lines)} lines")#!/usr/bin/env python3
import os
import shutil
import sys
import argparse
from pathlib import Path
from read_write_model import *

def process_folders(base_path, folder_type):
    """Process image or mask folders - delete and rename as specified"""
    path = Path(base_path) / folder_type
    
    if not path.exists() or not path.is_dir():
        print(f"{path} does not exist or is not a directory")
        return
    
    # Folders to delete
    for folder_name in ["cam2", "cam4", "cam6", "cam8"]:
        folder_path = path / folder_name
        if folder_path.exists():
            print(f"Deleting {folder_path}")
            shutil.rmtree(folder_path)
    
    # Rename folders
    rename_map = {
        "cam3": "cam2",
        "cam5": "cam3",
        "cam7": "cam4",
        "cam9": "cam5",
        "cam10": "cam6"
    }
    
    # Create a list of temporary names first to avoid conflicts during renaming
    for src, dst in rename_map.items():
        src_path = path / src
        if src_path.exists():
            temp_path = path / f"temp_{src}"
            print(f"Renaming {src_path} to {temp_path} (temporary)")
            src_path.rename(temp_path)
    
    # Now rename from temporary names to final names
    for src, dst in rename_map.items():
        temp_path = path / f"temp_{src}"
        if temp_path.exists():
            dst_path = path / dst
            print(f"Renaming {temp_path} to {dst_path} (final)")
            temp_path.rename(dst_path)

def process_colmap_model(model_path):
    """Process COLMAP model files - update images.bin"""
    if not Path(model_path).exists():
        print(f"Model path {model_path} does not exist")
        return
    
    # Read the COLMAP model
    images = read_images_binary(os.path.join(model_path, "images.bin"))
    
    # Create a new images dictionary
    new_images = {}
    new_image_id = 1  # Start with image ID 1
    
    # Process each image in the model
    for _, image in images.items():
        image_name = image.name
        
        # Check if the image contains any of the cameras to delete
        if any(f"cam{cam}" in image_name for cam in [2, 4, 6, 8]):
            print(f"Deleting image {image_name} from model")
            continue
        
        # Rename the image if needed
        new_name = image_name
        for src, dst in [("cam3", "cam2"), ("cam5", "cam3"), ("cam7", "cam4"), ("cam9", "cam5"), ("cam10", "cam6")]:
            if src in image_name:
                new_name = image_name.replace(src, dst)
                print(f"Renaming image in model: {image_name} -> {new_name}")
                break
        
        # Create a new Image object with the updated name and ID
        new_image = Image(
            id=new_image_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=new_name,
            xys=image.xys,
            point3D_ids=image.point3D_ids
        )
        
        new_images[new_image_id] = new_image
        new_image_id += 1
    
    # Write the updated images back
    write_images_binary(new_images, os.path.join(model_path, "images.bin"))
    
    print(f"COLMAP model updated at {model_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Modify camera calibration repository structure')
    parser.add_argument('--project_dir', required=True, help='Path to the project directory')
    args = parser.parse_args()
    
    project_dir = args.project_dir
    
    # Paths for rectified folders
    rectified_path = os.path.join(project_dir, "camera_calibration", "rectified")
    
    # Process image folders
    process_folders(rectified_path, "images")
    
    # Process mask folders if they exist
    process_folders(rectified_path, "masks")
    
    # Process COLMAP model in rectified/sparse
    process_colmap_model(
        os.path.join(rectified_path, "sparse")
    )
    
    # Process COLMAP model in aligned/sparse/0
    aligned_model_path = os.path.join(project_dir, "camera_calibration", "aligned", "sparse", "0")
    process_colmap_model(aligned_model_path)
    
    test_file_path = os.path.join(project_dir, "camera_calibration", "extras")
    # Process test.txt file
    process_test_file(test_file_path)
    
    print("All modifications completed successfully")

if __name__ == "__main__":
    main()