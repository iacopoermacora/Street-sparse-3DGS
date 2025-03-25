import cv2
import numpy as np
import os
import argparse
import json
import struct
import collections
from pathlib import Path
import re
import sys
from read_write_model import *

def convert_depth_map_to_meters(depth_image):
    """
    Convert Cyclomedia-encoded depth map to floating point distances in meters.
    """
    # Extract the G and R channels
    if depth_image.ndim == 3:
        b, g, r = cv2.split(depth_image)
    else:
        # If grayscale, assume it's already processed
        return depth_image.astype(np.float32)
    
    # Create a mask for background pixels (completely black in all channels)
    background_mask = (b == 0) & (g == 0) & (r == 0)
    
    r = r.astype(np.int64)
    g = g.astype(np.int64)

    precision = np.left_shift(np.right_shift(r, 6), 1)
    units = np.bitwise_or(np.left_shift(np.bitwise_and(r, 63), 8), g)
    depth_in_mm = np.left_shift(units, precision)
    depth_in_mm_f = depth_in_mm.astype(np.float64)
    depth_in_m_f = np.divide(depth_in_mm_f, 1000)
    
    # Set background pixels to zero
    depth_in_m_f[background_mask] = 0
    
    return depth_in_m_f, background_mask

def convert_to_gaussian_splatting_format(depth_in_meters, background_mask, min_depth=0.1, max_depth=None):
    """
    Convert depth in meters to the inverse depth format used by 3D Gaussian Splatting.
    Preserves completely black pixels (background).
    """
    # Create a mask for valid depth values (not background and above min_depth)
    valid_mask = (depth_in_meters > min_depth) & (~background_mask)
    
    if max_depth is not None:
        valid_mask &= depth_in_meters < max_depth
    
    # Calculate inverse depth (1/depth)
    inv_depth = np.zeros_like(depth_in_meters)
    inv_depth[valid_mask] = 1.0 / depth_in_meters[valid_mask]
    
    # Normalize to [0, 1] range (only for valid pixels)
    if np.sum(valid_mask) > 0:
        inv_depth_min = np.min(inv_depth[valid_mask])
        inv_depth_max = np.max(inv_depth[valid_mask])
        
        # Avoid division by zero
        if inv_depth_max > inv_depth_min:
            # Only normalize non-background pixels
            inv_depth_norm = np.zeros_like(inv_depth)
            inv_depth_norm[valid_mask] = (inv_depth[valid_mask] - inv_depth_min) / (inv_depth_max - inv_depth_min)
        else:
            inv_depth_norm = np.zeros_like(inv_depth)
    else:
        inv_depth_norm = np.zeros_like(inv_depth)
    
    # Convert to 16-bit unsigned integer
    depth_16bit = (inv_depth_norm * 65535).astype(np.uint16)
    
    # Ensure background pixels stay black (0)
    depth_16bit[background_mask] = 0
    
    return depth_16bit

def read_colmap_data(images_bin_path, images_depths_bin_path=None):
    """
    Read COLMAP data using the read_write_model module.
    Returns a dictionary mapping image IDs to their data.
    """
    images_data = {}
    
    try:
        # Use read_model to read the data
        if os.path.exists(images_bin_path):
            images = read_images_binary(images_bin_path)
            
            # Convert to our format
            for image_id, image in images.items():
                images_data[image_id] = {
                    'name': image.name,
                    'camera_id': image.camera_id
                }
        
        # Add depths data if provided
        if images_depths_bin_path and os.path.exists(images_depths_bin_path):
            depths_images = read_images_binary(images_depths_bin_path)
                
            for image_id_depths, image_depths in depths_images.items():
                images_data[image_id_depths+image_id+1] = {
                    'name': image_depths.name,
                    'camera_id': image_depths.camera_id
                }
    except Exception as e:
        print(f"Error reading COLMAP data: {e}")
        
    return images_data

def extract_image_info(original_filename):
    """
    Extract imageID and face information from the original filename.
    Original format: imageID_faceNumber_UppercaseFaceLetter_0_0.png
    e.g., WE931VUJ_1_B_0_0.png
    
    Returns: imageID, faceNumber, faceLetter
    """
    parts = original_filename.split('_')
    if len(parts) >= 3:
        image_id = parts[0]
        face_number = parts[1]
        face_letter = parts[2]
        return image_id, face_number, face_letter
    else:
        return None, None, None

def create_mapping_from_colmap(images_bin_path, images_depths_bin_path=None):
    """
    Create a mapping from original filename pattern to COLMAP naming.
    Original format: imageID_faceNumber_UppercaseFaceLetter_0_0.png
    COLMAP format: camX/increasingNumber_imageID_lowercaseFaceLetterFaceNumber.jpg
    
    Returns a dictionary that maps (imageID, faceNumber, faceLetter) to COLMAP name
    """
    try:
        colmap_images = read_colmap_data(images_bin_path, images_depths_bin_path)
        
        # Create mapping from original naming to COLMAP naming
        mapping = {}
        
        for image_id, data in colmap_images.items():
            colmap_name = data['name']
            
            # Extract pattern from COLMAP name: camX/increasingNumber_imageID_lowercaseFaceLetterFaceNumber.jpg
            # Handle the possibility of multi-digit face numbers (e.g., b10)
            match = re.search(r'cam\d+/\d+_([A-Z0-9]+)_([a-z])(\d+)\.jpg', colmap_name)
            if match:
                image_id_colmap = match.group(1)
                face_letter_colmap_lower = match.group(2)
                face_number_colmap = match.group(3)
                
                # Create the key for the mapping based on expected original format
                # Convert lowercase face letter to uppercase for matching with original files
                face_letter_upper = face_letter_colmap_lower.upper()
                
                key = (image_id_colmap, face_number_colmap, face_letter_upper)
                mapping[key] = colmap_name
        
        return mapping
    except Exception as e:
        print(f"Error creating mapping: {e}")
        return {}

def get_output_filename(original_filename, mapping):
    """
    Get the output filename for a depth map based on the original filename and the mapping.
    Changes extension from .jpg to .png for depth maps.
    """
    image_id, face_number, face_letter = extract_image_info(original_filename)
    
    if (image_id, face_number, face_letter) in mapping:
        colmap_name = mapping[(image_id, face_number, face_letter)]
        # Change extension to .png for depth maps
        return colmap_name.replace('.jpg', '.png')
    else:
        print(f"Not present in mapping: {image_id}, {face_number}, {face_letter}")
    
    return None

def process_depth_maps(project_dir, min_depth=0.1, max_depth=None):
    """
    Process all depth maps, rename them according to COLMAP naming, and save them in the right structure.
    
    Args:
        project_dir: Root project directory containing the COLMAP data
        min_depth: Minimum depth value to consider
        max_depth: Maximum depth value to consider
    """
    # Define paths
    colmap_images_bin = os.path.join(project_dir, "camera_calibration", "aligned", "sparse", "0", "images.bin")
    colmap_images_depths_bin = os.path.join(project_dir, "camera_calibration", "aligned", "sparse", "0", "images_depths.bin")
    input_depth_dir = os.path.join(project_dir, "camera_calibration", "depth_files", "rgb_depths")  # Assuming this is where your original depth maps are
    output_depth_dir = os.path.join(project_dir, "camera_calibration", "rectified", "depths")
    
    # Ensure input_depth_dir exists
    if not os.path.exists(input_depth_dir):
        print(f"Error: Input depth directory {input_depth_dir} does not exist")
        return
    
    # Create output directory structure
    os.makedirs(output_depth_dir, exist_ok=True)
    
    # Create mapping from original filenames to COLMAP names
    try:
        mapping = create_mapping_from_colmap(colmap_images_bin, colmap_images_depths_bin)
        print(f"Created mapping for {len(mapping)} images")
    except Exception as e:
        print(f"Error reading COLMAP files: {e}")
        return
    
    # Get list of all PNG files in the input directory
    input_files = [f for f in os.listdir(input_depth_dir) if f.lower().endswith('.png')]
    print(f"Found {len(input_files)} depth maps to process")
    
    # Track processed files for reporting
    processed_files = 0
    skipped_files = 0
    
    for i, filename in enumerate(input_files):
        print(f"Processing {i+1}/{len(input_files)}: {filename}")
        
        # Get the output filename based on COLMAP naming
        output_filename = get_output_filename(filename, mapping)
        if output_filename is None:
            print(f"  Skipping {filename}: No matching COLMAP image found")
            skipped_files += 1
            continue
        
        # Read the input depth map
        input_path = os.path.join(input_depth_dir, filename)
        depth_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if depth_image is None:
            print(f"  Error: Could not read {input_path}")
            skipped_files += 1
            continue
        
        # Convert to meters and get background mask
        depth_in_meters, background_mask = convert_depth_map_to_meters(depth_image)
        
        # Convert to Gaussian Splatting format
        depth_16bit = convert_to_gaussian_splatting_format(depth_in_meters, background_mask, min_depth, max_depth)
        
        # Determine output path including camera directory
        cam_dir = os.path.dirname(output_filename)
        full_cam_dir = os.path.join(output_depth_dir, cam_dir)
        os.makedirs(full_cam_dir, exist_ok=True)
        
        # Save the converted depth map
        output_path = os.path.join(output_depth_dir, output_filename)
        cv2.imwrite(output_path, depth_16bit)
        
        processed_files += 1
        print(f"  Saved to {output_path}")
    
    print(f"Processed {processed_files} depth maps, skipped {skipped_files} depth maps")

def find_chunks(project_dir):
    """
    Find all chunks in the project directory.
    Returns a list of chunk directories.
    """
    chunks_dir = os.path.join(project_dir, "camera_calibration", "chunks")
    if not os.path.exists(chunks_dir):
        return []
    
    chunks = []
    for item in os.listdir(chunks_dir):
        chunk_path = os.path.join(chunks_dir, item)
        if os.path.isdir(chunk_path):
            sparse_dir = os.path.join(chunk_path, "sparse", "0")
            if os.path.exists(sparse_dir):
                chunks.append(chunk_path)
    
    return chunks

def create_depth_params_json_for_chunk(chunk_dir):
    """
    Create a depth_params.json file for a specific chunk.
    
    Args:
        chunk_dir: Path to the chunk directory
    """
    sparse_dir = os.path.join(chunk_dir, "sparse", "0")
    images_bin_path = os.path.join(sparse_dir, "images.bin")
    images_depths_bin_path = os.path.join(sparse_dir, "images_depths.bin")
    
    if not os.path.exists(images_bin_path) and not os.path.exists(images_depths_bin_path):
        print(f"Skipping chunk {os.path.basename(chunk_dir)}: Neither images.bin nor images_depths.bin found")
        return 0
    
    # Read image data from both COLMAP files
    try:
        image_data = read_colmap_data(images_bin_path, images_depths_bin_path)
    except Exception as e:
        print(f"Error reading COLMAP data for chunk {os.path.basename(chunk_dir)}: {e}")
        return 0
    
    # Create depth parameters for all images
    depth_params = {}
    for image_id, data in image_data.items():
        # Get the image name without extension
        image_name = os.path.splitext(data['name'])[0]
        depth_params[image_name] = {"scale": 1.0, "offset": 0.0}
    
    # Save the depth_params.json file
    depth_params_path = os.path.join(sparse_dir, "depth_params.json")
    with open(depth_params_path, "w") as f:
        json.dump(depth_params, f, indent=2)
    
    print(f"Created depth_params.json for chunk {os.path.basename(chunk_dir)} with {len(depth_params)} entries")
    return len(depth_params)

def create_depth_params_for_all_chunks(project_dir):
    """
    Create depth_params.json files for all chunks in the project.
    
    Args:
        project_dir: Root project directory
    """
    chunks = find_chunks(project_dir)
    
    if not chunks:
        print("No chunks found in the project directory")
        return
    
    total_entries = 0
    for chunk_dir in chunks:
        print(f"Processing chunk: {os.path.basename(chunk_dir)}")
        entries = create_depth_params_json_for_chunk(chunk_dir)
        total_entries += entries
    
    print(f"Created depth_params.json files for {len(chunks)} chunks with a total of {total_entries} entries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert depth maps to 3D Gaussian Splatting format with COLMAP integration")
    parser.add_argument("--project_dir", required=True, help="Root project directory")
    parser.add_argument("--min_depth", type=float, default=0.1, help="Minimum depth value to consider")
    parser.add_argument("--max_depth", type=float, default=None, help="Maximum depth value to consider")
    
    args = parser.parse_args()
    
    # Process the depth maps
    process_depth_maps(args.project_dir, args.min_depth, args.max_depth)
    
    # Create depth_params.json files for all chunks
    create_depth_params_for_all_chunks(args.project_dir)