#!/usr/bin/env python3
"""
LAZ to PLY Point Cloud Converter with Translation

This script processes a folder of LAZ files into a combined PLY point cloud,
applies translation to align coordinate systems, and converts it to float precision.

Author: Based on work by Iacopo Ermacora
"""

import os
import time
import argparse
import json
import numpy as np
import laspy
import open3d as o3d
import trimesh
from tqdm import tqdm


def convert_ply_to_float(ply_path):
    """
    Loads a PLY file using trimesh, converts its vertex positions
    from double to float (32-bit) and re-exports it.
    """
    try:
        # Load without processing to preserve original data layout
        cloud = trimesh.load(ply_path, process=False)
        if hasattr(cloud, 'vertices'):
            cloud.vertices = cloud.vertices.astype(np.float32)
            cloud.export(ply_path)
            print(f"Converted {ply_path} vertices to float32.")
        else:
            print(f"No vertices found in {ply_path}.")
    except Exception as e:
        print(f"Error converting {ply_path}: {e}")


def laz_to_o3d_pointcloud(laz_filepath, translation=None):
    """
    Converts a LAZ file to an Open3D PointCloud with optional translation.
    
    Args:
        laz_filepath (str): Path to the LAZ file
        translation (dict, optional): Dict with 'x_translation' and 'y_translation' values
    
    Returns:
        o3d.geometry.PointCloud: The converted point cloud
    """
    print(f"Reading {laz_filepath} and converting to Open3D PointCloud")
    try:
        lasf = laspy.read(laz_filepath)
        points = np.vstack((lasf.x, lasf.y, lasf.z)).transpose().astype(np.float32)
        
        # Apply translation if provided
        if translation is not None:
            points[:, 0] = points[:, 0] - translation.get('x_translation', 0)
            points[:, 1] = points[:, 1] - translation.get('y_translation', 0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Check for color channels
        color_channels = {}
        for dim in lasf.header.point_format.dimensions:
            name = dim.name.lower()
            if name in ['red', 'green', 'blue']:
                color_channels[name] = name

        if len(color_channels) == 3:
            colors = np.vstack((lasf[color_channels['red']], 
                             lasf[color_channels['green']], 
                             lasf[color_channels['blue']])).transpose()
            colors = colors.astype(np.float32) / 65535.0  # Normalize 16-bit to [0,1]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        print(f"File converted with {len(pcd.points)} points")
        return pcd
    
    except Exception as e:
        print(f"Error reading {laz_filepath}: {e}")
        return None


def process_laz_folder(input_folder, output_path, translation_file=None):
    """
    Process all LAZ files in a folder into a single PLY point cloud.
    
    Args:
        input_folder (str): Path to the folder containing LAZ files
        output_path (str): Path where to save the resulting PLY file
        translation_file (str, optional): Path to a JSON file with translation values
        
    Returns:
        o3d.geometry.PointCloud: The combined point cloud
    """
    start_time = time.time()
    
    # Load translation if provided
    translation = None
    if translation_file and os.path.exists(translation_file):
        try:
            with open(translation_file, 'r') as f:
                translation = json.load(f)
            print(f"Loaded translation: x={translation.get('x_translation', 0)}, y={translation.get('y_translation', 0)}")
        except Exception as e:
            print(f"Error loading translation file: {e}")
    
    # Find all LAZ files in the input directory
    laz_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                if f.lower().endswith(".laz")]
    
    if not laz_files:
        print(f"No LAZ files found in {input_folder}")
        return None
    
    print(f"Found {len(laz_files)} LAZ files in {input_folder}")
    
    # Process each LAZ file and combine point clouds
    combined_pcd = o3d.geometry.PointCloud()
    for laz_file in tqdm(laz_files):
        pcd = laz_to_o3d_pointcloud(laz_file, translation)
        if pcd is None or len(pcd.points) == 0:
            print(f"Skipping {laz_file}")
            continue
        combined_pcd += pcd
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the combined point cloud
    print(f"Saving combined point cloud with {len(combined_pcd.points)} points to {output_path}")
    o3d.io.write_point_cloud(output_path, combined_pcd)
    
    # Convert the saved ply file from double to float using trimesh
    convert_ply_to_float(output_path)
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    return combined_pcd


def main():
    parser = argparse.ArgumentParser(description="Convert LAZ files to a combined PLY point cloud")
    parser.add_argument('--input_folder', type=str, required=True, 
                        help="Directory containing LAZ files")
    parser.add_argument('--output_path', type=str, required=True, 
                        help="Path for the output PLY file")
    parser.add_argument('--translation_file', type=str, default=None,
                        help="Path to JSON file with x_translation and y_translation values")
    
    args = parser.parse_args()
    
    process_laz_folder(args.input_folder, args.output_path, args.translation_file)


if __name__ == "__main__":
    main()