import cv2
import numpy as np
import os
import argparse
import json
from pathlib import Path

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
    
    Args:
        depth_in_meters: Depth map in meters
        background_mask: Boolean mask identifying background pixels
        min_depth: Minimum depth value to consider (helps avoid division by zero)
        max_depth: Maximum depth value to consider (helps with normalization)
    
    Returns:
        16-bit grayscale image with inverse depth values
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

def process_directory(input_dir, output_dir, min_depth=0.1, max_depth=None):
    """
    Process all depth maps in a directory and save them in the format required by 3D Gaussian Splatting.
    Preserves completely black pixels (background).
    
    Args:
        input_dir: Directory containing the input depth maps
        output_dir: Directory to save the converted depth maps
        min_depth: Minimum depth value to consider
        max_depth: Maximum depth value to consider
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all PNG files in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    
    print(f"Found {len(input_files)} depth maps to process")
    
    for i, filename in enumerate(input_files):
        print(f"Processing {i+1}/{len(input_files)}: {filename}")
        
        # Read the input depth map
        input_path = os.path.join(input_dir, filename)
        depth_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if depth_image is None:
            print(f"  Error: Could not read {input_path}")
            continue
        
        # Convert to meters and get background mask
        depth_in_meters, background_mask = convert_depth_map_to_meters(depth_image)
        
        # Convert to Gaussian Splatting format
        depth_16bit = convert_to_gaussian_splatting_format(depth_in_meters, background_mask, min_depth, max_depth)
        
        # Save the converted depth map
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, depth_16bit)
        
        print(f"  Saved to {output_path}")

def create_depth_params_json(output_dir, chunks_dir):
    """
    Create a depth_params.json file with scale=1.0 and offset=0.0 for all images.
    
    Args:
        output_dir: Directory where the depth_params.json file will be saved
        chunks_dir: Directory containing the chunks
    """
    depth_params = {}
    
    # Get list of all image files in the chunks directory
    for chunk_dir in os.listdir(chunks_dir):
        chunk_path = os.path.join(chunks_dir, chunk_dir)
        if not os.path.isdir(chunk_path):
            continue
        
        images_dir = os.path.join(chunk_path, "images")
        if not os.path.exists(images_dir):
            continue
        
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Get the base name without extension
                base_name = os.path.splitext(filename)[0]
                depth_params[base_name] = {"scale": 1.0, "offset": 0.0}
    
    # Save the depth_params.json file
    with open(os.path.join(output_dir, "depth_params.json"), "w") as f:
        json.dump(depth_params, f, indent=2)
    
    print(f"Created depth_params.json with {len(depth_params)} entries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert depth maps to 3D Gaussian Splatting format")
    parser.add_argument("--input_dir", required=True, help="Directory containing input depth maps")
    parser.add_argument("--output_dir", required=True, help="Directory to save converted depth maps")
    parser.add_argument("--min_depth", type=float, default=0.1, help="Minimum depth value to consider")
    parser.add_argument("--max_depth", type=float, default=None, help="Maximum depth value to consider")
    parser.add_argument("--chunks_dir", help="Directory containing chunks for creating depth_params.json")
    
    args = parser.parse_args()
    
    # Process the depth maps
    process_directory(args.input_dir, args.output_dir, args.min_depth, args.max_depth)
    
    # Create depth_params.json if chunks_dir is provided
    if args.chunks_dir:
        create_depth_params_json(args.output_dir, args.chunks_dir)