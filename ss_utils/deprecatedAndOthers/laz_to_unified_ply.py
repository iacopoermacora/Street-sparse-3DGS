#!/usr/bin/env python3
import os
import argparse
import numpy as np
import laspy
import open3d as o3d
from tqdm import tqdm

def laz_to_ply_converter(input_folder, output_file, x_translation=-136757, y_translation=-455750):
    """
    Convert multiple LAZ files from a folder into a single unified PLY file,
    applying translation to coordinates. Ensures single-precision output.
    
    Args:
        input_folder (str): Path to the folder containing LAZ files.
        output_file (str): Path to the output PLY file.
        x_translation (float): Translation to apply to X coordinates.
        y_translation (float): Translation to apply to Y coordinates.
    """
    # Check if input folder exists
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    # Get all LAZ files from the input folder
    laz_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.laz')]
    
    if not laz_files:
        raise ValueError(f"No LAZ files found in the input folder: {input_folder}")
    
    print(f"Found {len(laz_files)} LAZ files in the input folder.")
    print(f"Applying translation: X-{x_translation}, Y-{y_translation}")
    
    # Initialize empty arrays to store point cloud data
    all_points = []
    all_colors = []
    
    # Process each LAZ file
    for laz_file in tqdm(laz_files, desc="Processing LAZ files"):
        file_path = os.path.join(input_folder, laz_file)
        try:
            # Read LAZ file
            with laspy.open(file_path) as f:
                las = f.read()
                
                # Convert ScaledArrayView to numpy array first
                x_array = np.array(las.x) + x_translation  # Apply X translation
                y_array = np.array(las.y) + y_translation  # Apply Y translation
                z_array = np.array(las.z)
                
                # Convert to float32 (single-precision)
                x = x_array.astype(np.float32)
                y = y_array.astype(np.float32)
                z = z_array.astype(np.float32)
                
                points = np.vstack((x, y, z)).transpose()
                
                # Extract colors if available
                if all(hasattr(las, attr) for attr in ['red', 'green', 'blue']):
                    # Convert to numpy arrays first
                    red = np.array(las.red)
                    green = np.array(las.green)
                    blue = np.array(las.blue)
                    
                    # Normalize colors to [0, 1] range
                    red_norm = red / np.max(red) if np.max(red) > 0 else red
                    green_norm = green / np.max(green) if np.max(green) > 0 else green
                    blue_norm = blue / np.max(blue) if np.max(blue) > 0 else blue
                    
                    colors = np.vstack((red_norm, green_norm, blue_norm)).transpose().astype(np.float32)
                else:
                    # If colors are not available, use default color (gray)
                    colors = np.ones((points.shape[0], 3), dtype=np.float32) * 0.7
                
                all_points.append(points)
                all_colors.append(colors)
                
                print(f"Processed {laz_file}: {points.shape[0]} points")
                
        except Exception as e:
            print(f"Error processing file {laz_file}: {str(e)}")
    
    if not all_points:
        raise ValueError("No valid data found in any of the LAZ files.")
    
    # Concatenate all point cloud data
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    print(f"Merged point cloud contains {merged_points.shape[0]} points.")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    
    # Save as PLY file with explicit single-precision format
    o3d.io.write_point_cloud(output_file, pcd, write_ascii=True, compressed=False)
    print(f"Unified point cloud saved to {output_file}")
    
    # Verify file was created successfully
    if os.path.exists(output_file):
        print(f"Successfully created {output_file}")
    else:
        print("Warning: Output file was not created successfully.")

def main():
    parser = argparse.ArgumentParser(description="Convert folder of LAZ files to a unified PLY file")
    parser.add_argument("input_folder", help="Folder containing LAZ files")
    parser.add_argument("output_file", help="Output PLY file path")
    parser.add_argument("--x_translation", type=float, default=-136757, 
                        help="Translation to apply to X coordinates (default: -136757)")
    parser.add_argument("--y_translation", type=float, default=-455750, 
                        help="Translation to apply to Y coordinates (default: -455750)")
    args = parser.parse_args()
    
    laz_to_ply_converter(args.input_folder, args.output_file, 
                         args.x_translation, args.y_translation)

if __name__ == "__main__":
    main()