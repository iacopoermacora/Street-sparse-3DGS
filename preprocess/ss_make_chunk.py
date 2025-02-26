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

def process_lidar_points(input_files_path, output_file=None):
    """
    Reads and downsamples LiDAR points using voxel-based downsampling.
    Can save results to file and load from existing file if available.
    
    Args:
        input_files_path (str): Directory containing .laz files
        output_file (str, optional): Path to save/load the processed results
        
    Returns:
        tuple: (points, colors) where points is an Nx3 array of XYZ coordinates
               and colors is an Nx3 array of RGB values (0-255)
    """
    start_time = time.time()

    def downsample_for_max_density(pcd, max_density):
        """Downsamples a point cloud to ensure its density does not exceed max_density."""
        print(f"Downsampling point cloud to max density {max_density}")
        voxel_size = (1.0 / max_density) ** (1/3)
        return pcd.voxel_down_sample(voxel_size)

    def laz_to_o3d_pointcloud(laz_filepath):
        """Converts a LAZ file to an Open3D PointCloud."""
        print(f"Reading {laz_filepath} and converting to Open3D PointCloud")
        try:
            lasf = laspy.read(laz_filepath)
            points = np.vstack((lasf.x, lasf.y, lasf.z)).transpose()

            points[:, 0] = points[:, 0] - 136757
            points[:, 1] = points[:, 1] - 455750

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
            
            print(f"File converted. Loaded {len(pcd.points)} points")

            return pcd
        except Exception as e:
            print(f"Error reading {laz_filepath}: {e}")
            return None

    # Load pre-processed data if available
    if output_file and os.path.exists(output_file):
        print(f"Loading pre-processed data from {output_file}")
        data = np.load(output_file)
        return data['points'], data['colors']

    # Find all LAZ files in the input directory
    laz_files = [os.path.join(input_files_path, f) for f in os.listdir(input_files_path) 
                 if f.lower().endswith(".laz")]
    if not laz_files:
        print(f"No LAZ files found in {input_files_path}")
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # Process each LAZ file and combine downsampled point clouds
    combined_pcd = o3d.geometry.PointCloud()
    for laz_file in tqdm(laz_files):
        pcd = laz_to_o3d_pointcloud(laz_file)
        if pcd is None or len(pcd.points) == 0:
            continue
        
        # Downsample with target density of 8000 points per cubic unit
        downsampled_pcd = downsample_for_max_density(pcd, max_density=2000)
        combined_pcd += downsampled_pcd  # Combine into the aggregated point cloud

    # Extract points and colors
    if len(combined_pcd.points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    
    points = np.asarray(combined_pcd.points, dtype=np.float32)
    if combined_pcd.has_colors():
        colors = (np.asarray(combined_pcd.colors) * 255).astype(np.uint8)
    else:
        colors = np.zeros((len(points), 3), dtype=np.uint8)
    
    print(f"Total points after downsampling: {len(points)}")

    # Save results if requested
    if output_file:
        print(f"Saving processed data to {output_file}")
        np.savez_compressed(output_file, points=points, colors=colors)
    
    print("Processing LiDAR points took", time.time() - start_time)

    return points, colors

def process_lidar_for_chunk(i, j, corner_min, corner_max, lidar_xyz, lidar_rgb):
    """
    Filters the LiDAR points that belong to the given chunk.
    """
    mask = np.all(lidar_xyz < corner_max, axis=-1) & np.all(lidar_xyz > corner_min, axis=-1)
    return lidar_xyz[mask], lidar_rgb[mask]


def get_nb_pts(image_metas):
    n_pts = 0
    for key in image_metas:
        pts_idx = image_metas[key].point3D_ids
        if(len(pts_idx) > 5):
            n_pts = max(n_pts, np.max(pts_idx))

    return n_pts + 1

if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--chunk_size', default=100, type=float)
    parser.add_argument('--min_padd', default=0.2, type=float)
    parser.add_argument('--min_n_cams', default=50, type=int) # 100 NOTE: Changed from 100 to 50
    parser.add_argument('--max_n_cams', default=1500, type=int) # 1500
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--add_far_cams', default=True)
    parser.add_argument('--model_type', default="bin")

    args = parser.parse_args()

    # eval
    test_file = f"{args.base_dir}/test.txt"
    if os.path.exists(test_file):
        with open(test_file, 'r') as file:
            test_cam_names_list = file.readlines()
            blending_dict = {name[:-1] if name[-1] == '\n' else name: {} for name in test_cam_names_list}

    cam_intrinsics, images_metas, points3d = read_model(args.base_dir, ext=f".{args.model_type}")

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
        new_xyzs = xyzsC[mask]
        new_colors = colorsC[mask]
        new_indices = indicesC[mask]
        new_errors = errorsC[mask]

        new_colors = np.clip(new_colors, 0, 255).astype(np.uint8)

        valid_cam = np.all(cam_centers < corner_max, axis=-1) * np.all(cam_centers > corner_min, axis=-1)

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
                valid_cam[cam_idx] = n_pts > 50
            # If within 2x of the chunk
            elif np.all(cam_centers[cam_idx] < extended_corner_max) and np.all(cam_centers[cam_idx] > extended_corner_min):
                valid_cam[cam_idx] = n_pts > 50 and random.uniform(0, 1) > 0.5
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
        
        if valid_cam.sum() > args.min_n_cams:# or init_valid_cam.sum() > 0:
            out_path = os.path.join(args.output_path, f"{i}_{j}")
            out_colmap = os.path.join(out_path, "sparse", "0")
            os.makedirs(out_colmap, exist_ok=True)

            # must remove sfm points to use colmap triangulator in following steps
            images_out = {}
            for key in valid_keys:
                image_meta = images_metas[key]
                images_out[key] = Image(
                    id = key,
                    qvec = image_meta.qvec,
                    tvec = image_meta.tvec,
                    camera_id = image_meta.camera_id,
                    name = image_meta.name,
                    xys = [],
                    point3D_ids = []
                )

                if os.path.exists(test_file) and image_meta.name in blending_dict:
                    n_pts = np.isin(image_meta.point3D_ids, new_indices).sum()
                    blending_dict[image_meta.name][f"{i}_{j}"] = str(n_pts)

            print(f"Filtering the points for chunk {i}_{j}")
            # Filter LiDAR points for the chunk
            chunk_xyzs, chunk_colors = process_lidar_for_chunk(i, j, corner_min, corner_max, lidar_xyz, lidar_rgb)
            print(f"Filtered {len(chunk_xyzs)} LiDAR points out of {len(lidar_xyz)}")

            # Generate new unique IDs for points
            new_indices = np.arange(len(chunk_xyzs))

            points_out = {
                new_indices[idx] : Point3D(
                        id=new_indices[idx],
                        xyz=chunk_xyzs[idx],
                        rgb=chunk_colors[idx],
                        error=0.0,  # No error
                        image_ids=np.array([]),
                        point2D_idxs=np.array([]),
                    )
                for idx in range(len(chunk_xyzs))
            }

            write_model(cam_intrinsics, images_out, points_out, out_colmap, f".{args.model_type}")

            with open(os.path.join(out_path, "center.txt"), 'w') as f:
                f.write(' '.join(map(str, (corner_min + corner_max) / 2)))
            with open(os.path.join(out_path, "extent.txt"), 'w') as f:
                f.write(' '.join(map(str, corner_max - corner_min)))
        else:
            excluded_chunks.append([i, j])
            print("Chunk excluded")

    extent = global_bbox[1] - global_bbox[0]
    n_width = round(extent[0] / args.chunk_size)
    n_height = round(extent[1] / args.chunk_size)

    # Read LiDAR points once
    print("Reading LiDAR points")
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    LiDAR_path = f"{base_path}/ss_raw_images/LiDAR"
    lidar_xyz, lidar_rgb = process_lidar_points(LiDAR_path, f"{LiDAR_path}/downsampled.npz")  # Adjust path if necessary

    for i in range(n_width):
        for j in range(n_height):
            make_chunk(i, j, n_width, n_height)

    if os.path.exists(test_file):
        with open(f"{args.base_dir}/blending_dict.json", "w") as f:
            json.dump(blending_dict, f, indent=2)