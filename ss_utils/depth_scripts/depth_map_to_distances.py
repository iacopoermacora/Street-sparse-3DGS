'''
Thesis Project: Street-sparse-3DGS
Author: Iacopo Ermacora
Date: 11/2024-06/2025

Description: This script converts Cyclomedia-encoded depth maps to a format suitable 
for 3D Gaussian Splatting as required by the Hierarchical Gaussian Splatting pipeline.
'''
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
    The depth map is encoded in a 3-channel BGR format, where the R and G channels
    contain the depth information. The B channel is used to identify background pixels.

    Args:
        depth_image: Input depth image (BGR or grayscale).
    
    Returns:
        depth_in_m_f: Depth in meters as a float32 array.
        background_mask: Mask indicating background pixels (black).
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
    Convert depth in meters to the inverse depth format used by 3D Gaussian Splatting,
    and calculate the scale and offset needed to reconstruct the original inverse depth
    from the normalized 16-bit representation.
    Preserves completely black pixels (background).

    Args:
        depth_in_meters: Depth in meters as a float32 array.
        background_mask: Mask indicating background pixels (black).
        min_depth: Minimum depth value to consider.
        max_depth: Maximum depth value to consider.
    
    Returns:
        depth_16bit: Converted depth map in 16-bit unsigned integer format.
        scale: The scale factor (inv_depth_max - inv_depth_min) for depth_params.json.
        offset: The offset factor (inv_depth_min) for depth_params.json.
    """
    # Create a mask for valid depth values (not background and above min_depth)
    valid_mask = (depth_in_meters > min_depth) & (~background_mask)
    
    if max_depth is not None:
        valid_mask &= depth_in_meters < max_depth
    
    # Initialize scale and offset to safe values (indicating potentially invalid/uniform depth)
    scale = 0.0
    offset = 0.0
    
    # Calculate inverse depth (1/depth)
    inv_depth = np.zeros_like(depth_in_meters, dtype=np.float64) # Use float64 for precision

    if np.sum(valid_mask) > 0:
        inv_depth[valid_mask] = 1.0 / depth_in_meters[valid_mask]

        inv_depth_min = np.min(inv_depth[valid_mask])
        inv_depth_max = np.max(inv_depth[valid_mask])
        
        # Calculate scale and offset *before* normalization
        # Handle the case where max == min (uniform depth or single pixel)
        if inv_depth_max > inv_depth_min:
            scale = float(inv_depth_max - inv_depth_min) # Ensure float type
            offset = float(inv_depth_min)              # Ensure float type

            # Normalize to [0, 1] range (only for valid pixels)
            inv_depth_norm = np.zeros_like(inv_depth)
            # Use calculated scale (delta) to avoid potential floating point issues if max-min is tiny
            inv_depth_norm[valid_mask] = (inv_depth[valid_mask] - offset) / scale
        else:
            # If max == min, all valid pixels have the same inverse depth.
            # Normalization results in 0. Set scale to 0 and offset to the single value.
            inv_depth_norm = np.zeros_like(inv_depth)
            scale = 0.0
            offset = float(inv_depth_min) # The constant inverse depth value
    else:
        inv_depth_norm = np.zeros_like(inv_depth)

    # Clamp normalized values just in case of numerical instability
    inv_depth_norm = np.clip(inv_depth_norm, 0.0, 1.0)
    
    # Convert to 16-bit unsigned integer
    depth_16bit = (inv_depth_norm * 65535).astype(np.uint16)
    
    # Ensure background pixels stay black (0)
    depth_16bit[background_mask] = 0
    
    return depth_16bit, scale, offset

def read_colmap_data(images_bin_path, images_depths_bin_path=None):
    """
    Read COLMAP data using the read_write_model module.
    Returns a dictionary mapping image IDs to their data.

    Args:
        images_bin_path: Path to the images.bin file.
        images_depths_bin_path: Path to the images_depths.bin file (optional).
    
    Returns:
        images_data: Dictionary mapping image IDs to their data.
        depth_image_ids: Set of image IDs that come from images_depths.bin
    """
    images_data = {}
    depth_image_ids = set()  # Track which images come from images_depths.bin
    
    try:
        # Use read_model to read the data
        max_id = 0
        if os.path.exists(images_bin_path):
            images = read_images_binary(images_bin_path)
            print(f"Read {len(images)} images from images.bin")
            
            # Convert to our format
            for image_id, image in images.items():
                images_data[image_id] = {
                    'name': image.name,
                    'camera_id': image.camera_id,
                }
                max_id = max(max_id, image_id)
        
        # Add depths data if provided
        if images_depths_bin_path and os.path.exists(images_depths_bin_path):
            depths_images = read_images_binary(images_depths_bin_path)
            print(f"Read {len(depths_images)} images from images_depths.bin")
                
            for image_id_depths, image_depths in depths_images.items():
                # Ensure unique IDs by adding offset
                new_id = image_id_depths + max_id + 1
                images_data[new_id] = {
                    'name': image_depths.name,
                    'camera_id': image_depths.camera_id,
                }
                depth_image_ids.add(new_id)  # Add to depth image IDs set
    except Exception as e:
        print(f"Error reading COLMAP data: {e}")
        
    return images_data, depth_image_ids
    

def extract_image_info(original_filename):
    """
    Extract imageID and face information from the original filename.
    Original format: imageID_faceNumber_UppercaseFaceLetter_0_0.png
    e.g., WE931VUJ_1_B_0_0.png

    Args:
        original_filename: Original filename to extract information from.
    
    Returns:
        image_id: Extracted image ID.
        face_number: Extracted face number.
        face_letter: Extracted face letter.
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

    Args:
        images_bin_path: Path to the images.bin file.
        images_depths_bin_path: Path to the images_depths.bin file (optional).

    Returns:
        mapping: Dictionary mapping from original filenames to COLMAP filenames.
        depth_filenames: Set of filenames that are depth images.
    """
    try:
        colmap_images, depth_image_ids = read_colmap_data(images_bin_path, images_depths_bin_path)
        
        # Create mapping from original naming to COLMAP naming
        mapping = {}
        depth_filenames = set()  # Track which filenames are depth images
        
        for image_id, data in colmap_images.items():
            colmap_name = data['name']
            print(f"Processing COLMAP name: {colmap_name}")
            
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
                
                # If this is a depth image, add to the set of depth filenames
                if image_id in depth_image_ids:
                    depth_filenames.add(colmap_name)
        
        return mapping, depth_filenames
    except Exception as e:
        print(f"Error creating mapping: {e}")
        return {}, set()

def get_output_filename(original_filename, mapping):
    """
    Get the output filename for a depth map based on the original filename and the mapping.
    Changes extension from .jpg to .png for depth maps.

    Args:
        original_filename: Original filename to extract information from.
        mapping: Dictionary mapping from original filenames to COLMAP filenames.
    
    Returns:
        output_filename: Mapped filename for the depth map.
    """
    image_id, face_number, face_letter = extract_image_info(original_filename)
    
    if (image_id, face_number, face_letter) in mapping:
        colmap_name = mapping[(image_id, face_number, face_letter)]
        # Change extension to .png for depth maps
        return colmap_name.replace('.jpg', '.png')
    else:
        print(f"Not present in mapping: {image_id}, {face_number}, {face_letter}")
    
    return None

def create_white_mask(depth_image_shape):
    """
    Create a white mask image with the same shape as the depth image.
    
    Args:
        depth_image_shape: Shape of the depth image.
    
    Returns:
        white_mask: White mask image.
    """
    # Create a completely white image (all 255)
    white_mask = np.ones(depth_image_shape[:2], dtype=np.uint8) * 255
    return white_mask

def process_depth_maps(project_dir, min_depth=0.1, max_depth=None):
    """
    Process all depth maps, rename them according to COLMAP naming, 
    and save them in the right structure, and collect the calculated
    scale/offset parameters.
    
    Args:
        project_dir: Root project directory containing the COLMAP data
        min_depth: Minimum depth value to consider
        max_depth: Maximum depth value to consider
    
        Returns:
            calculated_depth_params: Dictionary mapping base image names to {"scale": scale, "offset": offset}
    """
    # Define paths
    colmap_images_bin = os.path.join(project_dir, "camera_calibration", "aligned", "sparse", "0", "images.bin")
    colmap_images_depths_bin = os.path.join(project_dir, "camera_calibration", "aligned", "sparse", "0", "images_depths.bin")
    input_depth_dir = os.path.join(project_dir, "camera_calibration", "depth_files", "rgb_depths")
    output_depth_dir = os.path.join(project_dir, "camera_calibration", "rectified", "depths")
    output_mask_dir = os.path.join(project_dir, "camera_calibration", "rectified", "masks")
    
    # Ensure input_depth_dir exists
    if not os.path.exists(input_depth_dir):
        print(f"Error: Input depth directory {input_depth_dir} does not exist")
        return
    
    # Create output directory structure
    os.makedirs(output_depth_dir, exist_ok=True)
    
    # Create mapping from original filenames to COLMAP names
    try:
        mapping, depth_filenames = create_mapping_from_colmap(colmap_images_bin, colmap_images_depths_bin)
        print(f"Created mapping for {len(mapping)} images")
    except Exception as e:
        print(f"Error reading COLMAP files: {e}")
        return {}

    # Dictionary to store calculated parameters
    calculated_depth_params = {}
    
    # Get list of all PNG files in the input directory
    input_files = [f for f in os.listdir(input_depth_dir) if f.lower().endswith('.png')]
    print(f"Found {len(input_files)} depth maps to process")
    
    # Track processed files for reporting
    processed_files = 0
    skipped_files = 0
    white_masks_created = 0
    
    for i, filename in enumerate(input_files):
        # Get the output filename based on COLMAP naming
        output_filename = get_output_filename(filename, mapping)
        if output_filename is None:
            print(f"  Skipping {filename}: No matching COLMAP image found")
            skipped_files += 1
            continue

        # Get the base name used as key in depth_params.json (e.g., camX/image)
        output_filename_base = os.path.splitext(output_filename)[0]
        
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
        depth_16bit, scale, offset = convert_to_gaussian_splatting_format(depth_in_meters, background_mask, min_depth, max_depth)
        
        # Store the calculated parameters
        calculated_depth_params[output_filename_base] = {"scale": scale, "offset": offset}

        # Determine output path including camera directory
        cam_dir = os.path.dirname(output_filename)
        full_cam_dir = os.path.join(output_depth_dir, cam_dir)
        os.makedirs(full_cam_dir, exist_ok=True)
        
        # Save the converted depth map
        output_path = os.path.join(output_depth_dir, output_filename)
        cv2.imwrite(output_path, depth_16bit)

        # If this image comes from images_depths.bin, create and save a white mask
        # Check if the COLMAP jpg name (before extension change) is in the depth_filenames set
        colmap_jpg_name = output_filename.replace('.png', '.jpg')
        if colmap_jpg_name in depth_filenames and os.path.exists(output_mask_dir):
            # Create white mask with same dimensions as depth image
            white_mask = create_white_mask(depth_image.shape)
            
            # Create mask directory structure
            mask_cam_dir = os.path.join(output_mask_dir, cam_dir)
            os.makedirs(mask_cam_dir, exist_ok=True)
            
            # Save white mask with same filename (but in masks directory)
            mask_path = os.path.join(output_mask_dir, output_filename)
            cv2.imwrite(mask_path, white_mask)
            white_masks_created += 1
        
        processed_files += 1
    
    print(f"Processed {processed_files} depth maps, skipped {skipped_files} depth maps")
    print(f"Created {white_masks_created} white masks (for depth only images)")

    return calculated_depth_params

def find_chunks(project_dir):
    """
    Find all chunks in the project directory.
    Returns a list of chunk directories.

    Args:
        project_dir: Root project directory
    
    Returns:
        List of chunk directories
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

def write_depth_params_for_model(model_dir, all_calculated_params, default_scale=0.0, default_offset=0.0):
    """
    Creates the depth_params.json file for a specific COLMAP model directory
    (e.g., a chunk or the aligned model) using pre-calculated scale/offset values.

    Args:
        model_dir: Path to the COLMAP model directory (containing images.bin/images_depths.bin).
        all_calculated_params: Dictionary mapping base image names to {"scale": scale, "offset": offset}.
        default_scale: Scale value to use if an image is not found in calculated_params.
        default_offset: Offset value to use if an image is not found in calculated_params.

    Returns:
        Number of entries written to the depth_params.json file.
    """
    sparse_dir = os.path.join(model_dir, "sparse", "0")
    images_bin_path = os.path.join(sparse_dir, "images.bin")
    images_depths_bin_path = os.path.join(sparse_dir, "images_depths.bin") # Optional

    # Check if at least one COLMAP image file exists
    if not os.path.exists(images_bin_path) and not os.path.exists(images_depths_bin_path):
        print(f"Skipping {os.path.basename(model_dir)}: Neither images.bin nor images_depths.bin found in {sparse_dir}")
        return 0

    print(f"Generating depth_params.json for: {sparse_dir}")

    # Read image data from COLMAP files to get the list of image names
    colmap_images = {}
    next_id = 0

    try:
        # Read images.bin and assign incremental IDs
        if os.path.exists(images_bin_path):
            images_bin_data = read_images_binary(images_bin_path)
            for _, image_data in images_bin_data.items():
                colmap_images[next_id] = image_data
                next_id += 1
            print(f" Read {len(images_bin_data)} images from images.bin")
            
        # Read images_depths.bin and continue with incremental IDs
        if os.path.exists(images_depths_bin_path):
            images_depths_data = read_images_binary(images_depths_bin_path)
            for _, image_data in images_depths_data.items():
                colmap_images[next_id] = image_data
                next_id += 1
            print(f" Read {len(images_depths_data)} images from images_depths.bin")
    except Exception as e:
        print(f" Error reading COLMAP image data for {os.path.basename(model_dir)}: {e}")
        return 0

    # Create depth parameters dictionary
    depth_params_output = {}
    found_count = 0
    not_found_count = 0

    for _, data in colmap_images.items():
        # Get the base image name (e.g., "cam0/00000_WE931VUJ_b1")
        image_name_base = os.path.splitext(data.name)[0]

        # Look up the calculated parameters
        params = all_calculated_params.get(image_name_base)

        if params is not None:
            depth_params_output[image_name_base] = params
            found_count += 1
        else:
            # Use default values if parameters were not calculated (e.g., depth file missing/error)
            depth_params_output[image_name_base] = {"scale": default_scale, "offset": default_offset}
            not_found_count += 1
            # print(f"  Warning: No calculated params found for {image_name_base}. Using defaults.")

    # Reorder the dictionary in alphabetical order of keys
    depth_params_output = dict(sorted(depth_params_output.items(), key=lambda x: x[0]))

    # Save the depth_params.json file
    depth_params_path = os.path.join(sparse_dir, "depth_params.json")
    try:
        with open(depth_params_path, "w") as f:
            json.dump(depth_params_output, f, indent=4) # Use indent 4 for readability
        print(f" Successfully wrote {len(depth_params_output)} entries to {depth_params_path}")
        if not_found_count > 0:
             print(f"   ({found_count} entries used calculated values, {not_found_count} used defaults (could be test images)")
    except Exception as e:
        print(f" Error writing {depth_params_path}: {e}")
        return 0

    return len(depth_params_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert depth maps to 3D Gaussian Splatting format with COLMAP integration")
    parser.add_argument("--project_dir", required=True, help="Root project directory")
    parser.add_argument("--min_depth", type=float, default=0.1, help="Minimum depth value to consider")
    parser.add_argument("--max_depth", type=float, default=None, help="Maximum depth value to consider")
    parser.add_argument("--default_scale", type=float, default=0.0, help="Scale value in depth_params.json if calculation fails")
    parser.add_argument("--default_offset", type=float, default=0.0, help="Offset value in depth_params.json if calculation fails")
    
    args = parser.parse_args()
    
    # Process the depth maps
    all_calculated_params = process_depth_maps(args.project_dir, args.min_depth, args.max_depth)
    
    aligned_model_dir = os.path.join(args.project_dir, "camera_calibration", "aligned")
    write_depth_params_for_model(aligned_model_dir, all_calculated_params, args.default_scale, args.default_offset)

    chunks = find_chunks(args.project_dir)
    print(f"\nProcessing {len(chunks)} Chunks...")
    total_chunk_entries = 0
    for chunk_dir in chunks:
        print(f" Processing Chunk: {os.path.basename(chunk_dir)}")
        entries = write_depth_params_for_model(chunk_dir, all_calculated_params, args.default_scale, args.default_offset)
        total_chunk_entries += entries
    print(f"Finished processing chunks. Total entries written: {total_chunk_entries}")