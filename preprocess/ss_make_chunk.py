'''
Thesis Project: Street-sparse-3DGS
Author: Iacopo Ermacora
Date: 11/2024-06/2025

Description: This script is used to create chunks of the 3D model from the COLMAP sparse 
reconstruction. It reads the COLMAP sparse reconstruction and LiDAR data, and creates chunks 
of the model based on the given parameters.
'''

#
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import os
import random
from read_write_model import *
import json
import laspy
from tqdm import tqdm
import open3d as o3d
import time
import trimesh

def process_lidar_points(input_files_path, center, extent, chunk_path):
    """
    Reads and downsamples LiDAR points using voxel-based downsampling.
    Can save results to file and load from existing file if available.
    
    Args:
        input_files_path (str): Directory containing .laz files
        center (tuple): (x, y) coordinates of the center of the chunk
        extent (tuple): (width, height) of the chunk
        chunk_path (str): Path to the chunk directory
        
    Returns:
        tuple: (points, colors) where points is an Nx3 array of XYZ coordinates
               and colors is an Nx3 array of RGB values (0-255)
    """
    start_time = time.time()

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

    def downsample_for_max_density(pcd, max_density):
        """Downsamples a point cloud to ensure its density does not exceed max_density."""
        print(f"Downsampling point cloud to max density {max_density}")
        voxel_size = (1.0 / max_density) ** (1/3)
        return pcd.voxel_down_sample(voxel_size)

    def laz_to_o3d_pointcloud(laz_filepath, translation, center, extent):
        """Converts a LAZ file to an Open3D PointCloud."""
        print(f"Reading {laz_filepath} and converting to Open3D PointCloud")
        try:
            lasf = laspy.read(laz_filepath)
            points = np.vstack((lasf.x, lasf.y, lasf.z)).transpose().astype(np.float32)

            points[:, 0] = points[:, 0] - translation['x_translation']
            points[:, 1] = points[:, 1] - translation['y_translation']

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
            
            print(f"Loaded {len(pcd.points)} points before masking")
            
            # Filter points outside the chunk
            mask = np.all(pcd.points < np.array([center[0] + extent[0]/2, center[1] + extent[1]/2, 1e12]), axis=-1) & \
                   np.all(pcd.points > np.array([center[0] - extent[0]/2, center[1] - extent[1]/2, -1e12]), axis=-1)
            # pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)[mask])
            
            # Get the indices of the points to keep
            indices = np.where(mask)[0]

            # Select the points and colors based on the indices
            pcd = pcd.select_by_index(indices)
            
            print(f"File converted. Loaded {len(pcd.points)} points")

            return pcd
        except Exception as e:
            print(f"Error reading {laz_filepath}: {e}")
            return None
    
    def check_overlap(chunk_center, chunk_extent, laz_file_origin, laz_file_extent):
        """
        Check if chunk and laz file overlap.
        
        Parameters:
        chunk_center (tuple): (x, y) coordinates of the center of the chunk
        chunk_extent (float): Width/height of the chunk
        laz_file_origin (tuple): (x, y) coordinates of the bottom-left corner of the LAZ file
        laz_file_extent (float): Width/height of the LAZ file
        
        Returns:
        bool: True if the two overlap, False otherwise
        """
        # Calculate the boundaries of square 1
        chunk_left = chunk_center[0] - chunk_extent/2
        chunk_right = chunk_center[0] + chunk_extent/2
        chunk_bottom = chunk_center[1] - chunk_extent/2
        chunk_top = chunk_center[1] + chunk_extent/2
        
        # Calculate the boundaries of square 2
        laz_left = laz_file_origin[0]
        laz_right = laz_file_origin[0] + laz_file_extent
        laz_bottom = laz_file_origin[1]
        laz_top = laz_file_origin[1] + laz_file_extent
        
        # Check for non-overlap conditions
        if chunk_right < laz_left or chunk_left > laz_right:
            return False  # No horizontal overlap
        if chunk_top < laz_bottom or chunk_bottom > laz_top:
            return False  # No vertical overlap
        
        # If we reach here, the squares overlap
        return True
    
    def filter_points_by_mesh_distance(pcd, mesh_path, max_distance=0.1):
        """
        Filter points in the point cloud that are within max_distance from the mesh.
        
        Args:
            pcd: The point cloud (o3d.geometry.PointCloud)
            mesh_path: Path to the reference mesh PLY file
            max_distance: Maximum distance threshold in meters (0.1 = 10 cm)
            
        Returns:
            filtered_pcd: Point cloud containing only points within the threshold
        """
        print(f"Filtering points by distance to mesh (max distance: {max_distance*100:.1f} cm)...")
        
        # Load the mesh
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if not mesh.has_triangles():
                print("Warning: The loaded mesh has no triangles. Skipping distance filtering.")
                return pcd
            
            # Ensure the mesh has vertex normals for distance computation
            if not mesh.has_vertex_normals():
                print("Computing mesh vertex normals...")
                mesh.compute_vertex_normals()
        except Exception as e:
            print(f"Error loading mesh from {mesh_path}: {e}")
            print("Skipping mesh-based filtering.")
            return pcd
        
        # Convert point cloud to numpy array for processing
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        
        print(f"Creating raycasting scene for {len(points)} points...")
        
        # Create a scene with the mesh for distance queries
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)
        
        # Process in batches to avoid memory issues
        batch_size = 100000
        n_points = len(points)
        n_batches = int(np.ceil(n_points / batch_size))
        
        # Lists to store filtered data
        filtered_indices = []
        
        print(f"Computing distances in {n_batches} batches...")
        
        # Process each batch
        for i in tqdm(range(n_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_points)
            
            # Get current batch
            batch_points = points[start_idx:end_idx]
            
            # Compute distances for the current batch
            query_points = o3d.core.Tensor(batch_points, dtype=o3d.core.Dtype.Float32)
            batch_distances = scene.compute_distance(query_points).numpy()
            
            # Filter based on distance threshold
            batch_mask = batch_distances <= max_distance
            
            # Store indices of points to keep
            batch_indices = np.where(batch_mask)[0] + start_idx
            filtered_indices.extend(batch_indices)
        
        # Create a new point cloud with the filtered points
        filtered_pcd = pcd.select_by_index(filtered_indices)
        
        print(f"Original point count: {len(pcd.points)}")
        print(f"Filtered point count: {len(filtered_pcd.points)}")
        print(f"Removed {len(pcd.points) - len(filtered_pcd.points)} points ({(1 - len(filtered_pcd.points)/len(pcd.points))*100:.2f}%)")
        
        return filtered_pcd

    # Find all LAZ files in the input directory
    laz_files = [os.path.join(input_files_path, f) for f in os.listdir(input_files_path) 
                 if f.lower().endswith(".laz")]
    if not laz_files:
        print(f"No LAZ files found in {input_files_path}")
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    
    # Filter the laz files to only include the ones that are in the chunk bounding box based on the center.txt and extent.txt files. 
    with open(f"{args.project_dir}/camera_calibration/translation.json", 'r') as f:
        translation = json.load(f)
    # Compute the tranlated position of the center of the chunk
    center_original = [0, 0]
    center_original[0] = center[0] + translation['x_translation']
    center_original[1] = center[1] + translation['y_translation']
    # Read the laz files and get the bounding box of the laz file
    laz_selected = []
    for laz_file in laz_files:
        # Get the x and y coordinates of the laz file from the file name
        x = int(laz_file.split('_')[-2])
        y = int(laz_file.split('_')[-1].split('.')[0])
        # Check if any point inside the bounding box of the chunk (center + extent) is inside the bounding box of the LiDAR file
        # If it is, add the laz file to the list of laz files to be processed
        # If it is not, skip the laz file
        if check_overlap(center_original, extent[0], (x*50, y*50), 50):
            laz_selected.append(laz_file)

    # Process each LAZ file and combine downsampled point clouds
    combined_pcd = o3d.geometry.PointCloud()
    for laz_file in tqdm(laz_selected):
        pcd = laz_to_o3d_pointcloud(laz_file, translation, center, extent)
        if pcd is None or len(pcd.points) == 0:
            print(f"Skipping {laz_file}")
            continue
        combined_pcd += pcd
    
    mesh_path = os.path.join(args.project_dir, "camera_calibration/depth_files/vis2mesh/total_vis2mesh.ply")
    # Filter points based on distance to mesh
    print("Filtering points based on distance to mesh...")
    combined_pcd = filter_points_by_mesh_distance(combined_pcd, mesh_path, max_distance=0.1)

    if args.LiDAR_initialisation:
        downsampled_pcd = downsample_for_max_density(combined_pcd, max_density=args.LiDAR_downsample_density)

        # Extract points and colors
        if len(downsampled_pcd.points) == 0:
            print("No points found in the LiDAR files")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        
        points = np.asarray(downsampled_pcd.points, dtype=np.float32)
        if downsampled_pcd.has_colors():
            colors = (np.asarray(downsampled_pcd.colors) * 255).astype(np.uint8)
        else:
            colors = np.zeros((len(points), 3), dtype=np.uint8)
        
        print(f"Total points after downsampling: {len(points)}")
    else:
        points = None
        colors = None
    
    # Save the non downsampled points as ply file
    ply_path = f"{chunk_path}/chunk.ply"
    o3d.io.write_point_cloud(ply_path, combined_pcd)
    
    # Convert the saved ply file from double to float using trimesh
    convert_ply_to_float(ply_path)
    
    print("Processing LiDAR points took", time.time() - start_time)

    return points, colors, combined_pcd

# def process_lidar_for_chunk(i, j, corner_min, corner_max, lidar_xyz, lidar_rgb):
#     """
#     Filters the LiDAR points that belong to the given chunk.
#     """
#     mask = np.all(lidar_xyz < corner_max, axis=-1) & np.all(lidar_xyz > corner_min, axis=-1)
#     return lidar_xyz[mask], lidar_rgb[mask]


def get_nb_pts(image_metas):
    n_pts = 0
    for key in image_metas:
        pts_idx = image_metas[key].point3D_ids
        if(len(pts_idx) > 5):
            n_pts = max(n_pts, np.max(pts_idx))

    return n_pts + 1

def fill_temporal_gaps_in_chunk(images_depths_out, corner_min, corner_max, args):
    """
    Fill temporal gaps in depth images by checking recording order and distance criteria.
    
    Args:
        images_depths_out: Dictionary of depth images already selected for the chunk
        corner_min, corner_max: Chunk boundaries
        args: Command line arguments containing project_dir
    
    Returns:
        Updated images_depths_out dictionary with gap-filled images
    """
    # Load recording details
    recording_details_path = os.path.join(args.project_dir, "camera_calibration/extras/recording_details_depths.json")
    if not os.path.exists(recording_details_path):
        print(f"Recording details file not found: {recording_details_path}")
        return images_depths_out
    
    with open(recording_details_path, 'r') as f:
        recording_data = json.load(f)
    
    # Extract and sort recording properties by time
    recording_properties = recording_data.get("RecordingProperties", [])
    recording_properties.sort(key=lambda x: x["RecordingTimeGps"])
    
    # Extract ImageIds from current chunk depth images (convert from colmap format)
    chunk_image_ids = []
    for key, image_meta in images_depths_out.items():
        # Extract ImageId from name like "camx/NUMBER_imageid_FACE.JPG"
        image_id = image_meta.name.split('/')[-1].split('_')[1]
        chunk_image_ids.append(image_id)
    
    if not chunk_image_ids:
        return images_depths_out
    
    # Find indices of chunk images in the sorted recording list
    chunk_indices = []
    for image_id in chunk_image_ids:
        for i, prop in enumerate(recording_properties):
            if prop["ImageId"] == image_id:
                chunk_indices.append(i)
                break
    
    if not chunk_indices:
        return images_depths_out
    
    chunk_indices.sort()
    
    def get_distance(prop1, prop2):
        """Calculate distance between two recording positions"""
        return np.sqrt((prop1["X"] - prop2["X"])**2 + (prop1["Y"] - prop2["Y"])**2)
    
    def add_depth_to_chunk(image_id):
        """Add an image to the chunk if it exists in images_depth_metas"""
        for key, image_meta in images_depth_metas.items():
            existing_image_id = image_meta.name.split('/')[-1].split('_')[1]
            if existing_image_id == image_id:
                images_depths_out[key] = Image(
                    id=key,
                    qvec=image_meta.qvec,
                    tvec=image_meta.tvec,
                    camera_id=image_meta.camera_id,
                    name=image_meta.name,
                    xys=image_meta.xys,
                    point3D_ids=image_meta.point3D_ids
                )
    
    # Check for gaps within the chunk sequence
    for i in range(len(chunk_indices) - 1):
        previous_idx = chunk_indices[i-1] if i > 0 else None
        current_idx = chunk_indices[i]
        next_idx = chunk_indices[i + 1]

        current_prop = recording_properties[current_idx]
        
        # Previous
        if previous_idx is not None:
            if current_idx - previous_idx > 1:
                current_prop = recording_properties[current_idx]
                actual_previous_prop = recording_properties[current_idx -1]

                # Check if distance between current and actual previous is less than 10m
                if get_distance(current_prop, actual_previous_prop) < 10.0:
                    add_depth_to_chunk(actual_previous_prop["ImageId"])
        
        # Next
        if next_idx - current_idx > 1:
            current_prop = recording_properties[current_idx]
            actual_next_prop = recording_properties[current_idx + 1]
            
            # Check if distance between current and actual next is less than 10m
            if get_distance(current_prop, actual_next_prop) < 10.0:
                add_depth_to_chunk(actual_next_prop["ImageId"])
    
    # Check image before the first chunk image
    if chunk_indices[0] > 0:
        first_prop = recording_properties[chunk_indices[0]]
        # Check if the index is valid
        if chunk_indices[0] - 1 >= 0:
            before_prop = recording_properties[chunk_indices[0] - 1]
        
            if get_distance(before_prop, first_prop) < 10.0:
                add_depth_to_chunk(before_prop["ImageId"])
    
    # Check image after the last chunk image
    if chunk_indices[-1] < len(recording_properties) - 1:
        last_prop = recording_properties[chunk_indices[-1]]

        # Check if the index is valid
        if chunk_indices[-1] + 1 < len(recording_properties):
            after_prop = recording_properties[chunk_indices[-1] + 1]
        
            if get_distance(last_prop, after_prop) < 10.0:
                add_depth_to_chunk(after_prop["ImageId"])
    
    return images_depths_out

def make_chunk(i, j, n_width, n_height):
    # in_path = f"{args.base_dir}/chunk_{i}_{j}"
    # if os.path.exists(in_path):
    print(f"chunk {i}_{j}")
    # corner_min, corner_max = bboxes[i, j, :, 0], bboxes[i, j, :, 1]
    corner_min = global_bbox[0] + np.array([i * args.chunk_size, j * args.chunk_size, 0])
    corner_max = global_bbox[0] + np.array([(i + 1) * args.chunk_size, (j + 1) * args.chunk_size, 1e12])
    corner_min[2] = -1e12
    corner_max[2] = 1e12
    
    corner_min_for_pts = corner_min.copy()
    corner_max_for_pts = corner_max.copy()
    if i == 0:
        corner_min_for_pts[0] = -1e12
    if j == 0:
        corner_min_for_pts[1] = -1e12
    if i == n_width - 1:
        corner_max_for_pts[0] = 1e12
    if j == n_height - 1:
        corner_max_for_pts[1] = 1e12

    mask = np.all(xyzsC < corner_max_for_pts, axis=-1) * np.all(xyzsC > corner_min_for_pts, axis=-1)
    colmap_xyzs = xyzsC[mask]
    colmap_colors = colorsC[mask]
    colmap_indices = indicesC[mask]
    colmap_errors = errorsC[mask]

    colmap_colors = np.clip(colmap_colors, 0, 255).astype(np.uint8)

    valid_cam = np.all(cam_centers < corner_max, axis=-1) * np.all(cam_centers > corner_min, axis=-1)

    print(f"Valid cameras before visibility-base selection: {valid_cam.sum()}")

    box_center = (corner_max + corner_min) / 2
    extent = (corner_max - corner_min) / 2
    acceptable_radius = 2
    extended_corner_min = box_center - acceptable_radius * extent
    extended_corner_max = box_center + acceptable_radius * extent

    for cam_idx, key in enumerate(images_metas):
        # if not valid_cam[cam_idx]:
        image_points3d =  images_points3d[key]
        n_pts = (np.all(image_points3d < corner_max_for_pts, axis=-1) * np.all(image_points3d > corner_min_for_pts, axis=-1)).sum() if len(image_points3d) > 0 else 0

        # If within chunk
        if np.all(cam_centers[cam_idx] < corner_max) and np.all(cam_centers[cam_idx] > corner_min):
            valid_cam[cam_idx] = True # PACOMMENT: Removed limit of 50 points because with our sparse data the points are not enough
        # If within 2x of the chunk
        elif np.all(cam_centers[cam_idx] < extended_corner_max) and np.all(cam_centers[cam_idx] > extended_corner_min):
            valid_cam[cam_idx] = n_pts > 20 # PACOMMENT: Removed limit of 50 points because with our sparse data the points are not enough
        # All distances
        if (not valid_cam[cam_idx]) and n_pts > 10 and args.add_far_cams:
            valid_cam[cam_idx] = random.uniform(0, 0.5) < (float(n_pts) / len(image_points3d))
        
    print(f"{valid_cam.sum()} valid cameras after visibility-base selection")

    if valid_cam.sum() > args.max_n_cams:
        for _ in range(valid_cam.sum() - args.max_n_cams):
            remove_idx = random.randint(0, valid_cam.sum() - 1)
            remove_idx_glob = np.arange(len(valid_cam))[valid_cam][remove_idx]
            valid_cam[remove_idx_glob] = False

        print(f"{valid_cam.sum()} after random removal")

    valid_keys = [key for idx, key in enumerate(images_metas) if valid_cam[idx]]
    
    if valid_cam.sum() > args.min_n_cams: # or init_valid_cam.sum() > 0:
        out_path = os.path.join(args.output_path, f"{i}_{j}")
        out_colmap = os.path.join(out_path, "sparse", "0")
        os.makedirs(out_colmap, exist_ok=True)

        # Create output images while keeping the original points
        images_out = {}
        for key in valid_keys:
            image_meta = images_metas[key]
        
            # Get the original points and their 2D locations
            pts_idx = image_meta.point3D_ids
            xys = image_meta.xys
            
            # Create a mask for valid points (non-negative IDs)
            valid_pts_mask = pts_idx >= 0

            # Initialize in_chunk_mask with all False
            in_chunk_mask = np.zeros_like(pts_idx, dtype=bool)
            
            # If there are valid points, check which ones are in this chunk
            if np.any(valid_pts_mask):
                # Get valid point IDs
                valid_point_ids = pts_idx[valid_pts_mask]
                
                # For each valid point ID, check if it's in the chunk
                for i_pt, pt_id in enumerate(valid_point_ids):
                    # Get the index in the original mask
                    orig_idx = np.where(valid_pts_mask)[0][i_pt]
                    
                    # If point exists in points3d
                    if pt_id in points3d:
                        pt_xyz = points3d[pt_id].xyz
                        # Check if point is in chunk
                        if (np.all(pt_xyz < corner_max_for_pts) and 
                            np.all(pt_xyz > corner_min_for_pts)):
                            in_chunk_mask[orig_idx] = True
            
            # Filter to keep only points in this chunk
            filtered_xys = xys[in_chunk_mask]
            filtered_point3D_ids = pts_idx[in_chunk_mask]
            
            images_out[key] = Image(
                id=key,
                qvec=image_meta.qvec,
                tvec=image_meta.tvec,
                camera_id=image_meta.camera_id,
                name=image_meta.name,
                xys=filtered_xys,
                point3D_ids=filtered_point3D_ids
            )

            if os.path.exists(test_file) and image_meta.name in blending_dict:
                n_pts = np.isin(image_meta.point3D_ids, colmap_indices).sum()
                blending_dict[image_meta.name][f"{i}_{j}"] = str(n_pts)
        
        images_depths_out = {}
        # Filter the images_depths to include only the ones in the chunk
        for key in images_depth_metas:
            # Get the camera center
            cam_center = -qvec2rotmat(images_depth_metas[key].qvec).T @ images_depth_metas[key].tvec
            # Check if the camera center is in the chunk
            if np.all(cam_center < corner_max) and np.all(cam_center > corner_min):
                # Add the image to the output
                images_depths_out[key] = Image(
                    id=key,
                    qvec=images_depth_metas[key].qvec,
                    tvec=images_depth_metas[key].tvec,
                    camera_id=images_depth_metas[key].camera_id,
                    name=images_depth_metas[key].name,
                    xys=images_depth_metas[key].xys,
                    point3D_ids=images_depth_metas[key].point3D_ids
                )
        
        # Fill temporal gaps in depth images
        images_depths_out = fill_temporal_gaps_in_chunk(images_depths_out, corner_min, corner_max, args)
        
        center = (corner_min + corner_max) / 2
        extent = corner_max - corner_min
        
        with open(os.path.join(out_path, "center.txt"), 'w') as f:
            f.write(' '.join(map(str, center)))
        with open(os.path.join(out_path, "extent.txt"), 'w') as f:
            f.write(' '.join(map(str, extent)))

        print(f"Filtering the points for chunk {i}_{j}")
        # Filter LiDAR points for the chunk
        print("Reading LiDAR points")
        LiDAR_path = f"{args.project_dir}/ss_raw_images/LiDAR"
        lidar_xyzs, lidar_colors, combined_pcd = process_lidar_points(LiDAR_path, center, extent, out_path)

        # Create points dictionary for output, including both COLMAP and LiDAR points
        points_out = {}
        
        # First add all the original COLMAP points
        for idx, original_id in enumerate(colmap_indices):
            if original_id in points3d:
                point = points3d[original_id]
                # Keep original point with all its properties
                points_out[original_id] = Point3D(
                    id=original_id,
                    xyz=colmap_xyzs[idx],
                    rgb=colmap_colors[idx],
                    error=colmap_errors[idx],
                    image_ids=point.image_ids,
                    point2D_idxs=point.point2D_idxs,
                )
        
        # Then add LiDAR points with new IDs that don't conflict
        if lidar_xyzs is not None and len(lidar_xyzs) > 0 and args.LiDAR_initialisation:
            # Find the maximum ID in the original points
            max_id = max(points3d.keys()) if points3d else 0
            
            # Add LiDAR points with new IDs starting after max_id
            for idx in range(len(lidar_xyzs)):
                new_id = max_id + idx + 1
                points_out[new_id] = Point3D(
                    id=new_id,
                    xyz=lidar_xyzs[idx],
                    rgb=lidar_colors[idx],
                    error=0.0,  # No error for LiDAR points
                    image_ids=np.array([]),  # No image associations
                    point2D_idxs=np.array([]),  # No image associations
                )
        
        print(f"Writing chunk {i}_{j} with {len(images_out)} images and {len(points_out)} points and {len(images_depths_out)} additional depth maps")

        write_model(cam_intrinsics, images_out, points_out, out_colmap, f".{args.model_type}")
        write_images_binary(images_depths_out, os.path.join(out_colmap, "images_depths.bin"))

    else:
        excluded_chunks.append([i, j])
        print("Chunk excluded")
        combined_pcd = None
        lidar_xyzs = None
        lidar_colors = None
    return combined_pcd, lidar_xyzs, lidar_colors

if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True)
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--chunk_size', default=100, type=float)
    parser.add_argument('--min_padd', default=0.2, type=float)
    parser.add_argument('--min_n_cams', default=5, type=int) # 100 NOTE: Changed from 100 to 5
    parser.add_argument('--max_n_cams', default=1500, type=int) # 1500
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--add_far_cams', default=True)
    parser.add_argument('--model_type', default="bin")

    parser.add_argument('--LiDAR_initialisation', action="store_true", default=False, help="Use this flag to initialise the point cloud with the LiDAR ground truth.")
    parser.add_argument('--LiDAR_downsample_density', type=int, default=2000, help="Downsample the LiDAR point cloud to this density. The density is in points per cubic meter.")

    args = parser.parse_args()

    # eval
    test_file = f"{args.base_dir}/test.txt"
    if os.path.exists(test_file):
        with open(test_file, 'r') as file:
            test_cam_names_list = file.readlines()
            blending_dict = {name[:-1] if name[-1] == '\n' else name: {} for name in test_cam_names_list}

    cam_intrinsics, images_metas, points3d = read_model(args.base_dir, ext=f".{args.model_type}")

    images_depth_metas = read_images_binary(os.path.join(args.base_dir, "images_depths.bin"))

    cam_centers = np.array([
        -qvec2rotmat(images_metas[key].qvec).astype(np.float32).T @ images_metas[key].tvec.astype(np.float32)
        for key in images_metas
    ])

    n_pts = get_nb_pts(images_metas)

    xyzs = np.zeros([n_pts, 3], np.float32)
    errors = np.zeros([n_pts], np.float32) + 9e9
    indices = np.zeros([n_pts], np.int64)
    n_images = np.zeros([n_pts], np.int64)
    colors = np.zeros([n_pts, 3], np.float32)

    idx = 0    
    for key in points3d:
        xyzs[idx] = points3d[key].xyz
        indices[idx] = points3d[key].id
        errors[idx] = points3d[key].error
        colors[idx] = points3d[key].rgb
        n_images[idx] = len(points3d[key].image_ids)
        idx +=1

    mask = errors < 1e1
    # mask *= n_images > 3
    xyzsC, colorsC, errorsC, indicesC, n_imagesC = xyzs[mask], colors[mask], errors[mask], indices[mask], n_images[mask]

    points3d_ordered = np.zeros([indicesC.max()+1, 3])
    points3d_ordered[indicesC] = xyzsC
    images_points3d = {}

    for key in images_metas:
        pts_idx = images_metas[key].point3D_ids
        mask = pts_idx >= 0
        mask *= pts_idx < len(points3d_ordered)
        pts_idx = pts_idx[mask]
        if len(pts_idx) > 0:
            image_points3d = points3d_ordered[pts_idx]
            mask = (image_points3d != 0).sum(axis=-1)
            # images_metas[key]["points3d"] = image_points3d[mask>0]
            images_points3d[key] = image_points3d[mask>0]
        else:
            # images_metas[key]["points3d"] = np.array([])
            images_points3d[key] = np.array([])


    global_bbox = np.stack([cam_centers.min(axis=0), cam_centers.max(axis=0)])
    global_bbox[0, :2] -= args.min_padd * args.chunk_size
    global_bbox[1, :2] += args.min_padd * args.chunk_size
    extent = global_bbox[1] - global_bbox[0]
    padd = np.array([args.chunk_size - extent[0] % args.chunk_size, args.chunk_size - extent[1] % args.chunk_size])
    global_bbox[0, :2] -= padd / 2
    global_bbox[1, :2] += padd / 2

    global_bbox[0, 2] = -1e12
    global_bbox[1, 2] = 1e12

    excluded_chunks = []
    chunks_pcd = {}
    
    extent = global_bbox[1] - global_bbox[0]
    n_width = round(extent[0] / args.chunk_size)
    n_height = round(extent[1] / args.chunk_size)

    total_pcd = o3d.geometry.PointCloud()
    total_xys = []
    total_colors = []
    for i in range(n_width):
        for j in range(n_height):
            chunk_pcd, chunk_xyzs, chunk_colors = make_chunk(i, j, n_width, n_height)
            if chunk_pcd is not None:
                total_pcd += chunk_pcd
                total_xys.append(chunk_xyzs)
                total_colors.append(chunk_colors)
    
    # Add to the original points3d the new points3d coming from the LiDAR without duplicating the ids
    if args.LiDAR_initialisation:
        print("Adding LiDAR points to the original points3D")
        # Calculate max_id once, outside the loops
        max_id = max(points3d.keys())
        
        # Counter to track total points added so far across all lists
        point_count = 0
        for i in range(len(total_xys)):
            if total_xys[i] is not None and len(total_xys[i]) > 0:
                for j in range(len(total_xys[i])):
                    new_id = max_id + point_count + 1
                    points3d[new_id] = Point3D(
                        id=new_id,
                        xyz=total_xys[i][j],
                        rgb=total_colors[i][j],
                        error=0.0,  # No error for LiDAR points
                        image_ids=np.array([]),  # No image associations
                        point2D_idxs=np.array([]),  # No image associations
                    )
                    point_count += 1
        # Rename the points3d file to points3d_uninitialised.bin
        if os.path.exists(f"{args.base_dir}/points3D.bin"):
            os.rename(f"{args.base_dir}/points3D.bin", f"{args.base_dir}/points3D_uninitialised.bin")

        # Save the new points3d to the base_dir in a points3D.bin file
        write_points3D_binary(points3d, f"{args.base_dir}/points3D.bin")

    if os.path.exists(test_file):
        with open(f"{args.base_dir}/blending_dict.json", "w") as f:
            json.dump(blending_dict, f, indent=2)