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

import os, sys
import subprocess
import argparse
import time, platform

def submit_job(slurm_args):
    """Submit a job using sbatch and return the job ID."""    
    try:
        result = subprocess.run(slurm_args, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error when submitting a job: {e}")
        sys.exit(1)
    
    job = result.stdout.strip().split()[-1]
    print(f"submitted job {job}")
    return job

def is_job_finished(job):
    """Check if the job has finished using sacct."""
    result = subprocess.run(['sacct', '-j', job, '--format=State', '--noheader', '--parsable2'], capture_output=True, text=True)
    
    job_state = result.stdout.split('\n')[0]
    return job_state if job_state in {'COMPLETED', 'FAILED', 'CANCELLED'} else ""

def setup_dirs(images, colmap, chunks, project):
    images_dir = os.path.join(project, "camera_calibration", "rectified", "images") if images == "" else images
    colmap_dir = os.path.join(project, "camera_calibration", "aligned") if colmap == "" else colmap
    chunks_dir = os.path.join(project, "camera_calibration") if chunks == "" else chunks

    return images_dir, colmap_dir, chunks_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', required=True, help="images, colmap and chunks paths doesnt have to be set if you generated the colmap using generate_colmap script.")
    parser.add_argument('--images_dir', default="")
    parser.add_argument('--global_colmap_dir', default="")
    parser.add_argument('--chunks_dir', default="")
    parser.add_argument('--use_slurm', action="store_true", default=False)
    parser.add_argument('--skip_bundle_adjustment', action="store_true", default=False)
    parser.add_argument('--n_jobs', type=int, default=8, help="Run per chunk COLMAP in parallel on the same machine. Does not handle multi GPU systems. --use_slurm overrides this.")

    # NOTE: Adding argument to deal with already generated colmap
    parser.add_argument('--LiDAR_initialisation', action="store_true", default=False, help="Use this flag to initialise the point cloud with the LiDAR ground truth.")
    parser.add_argument('--LiDAR_downsample_density', type=int, default=500, help="Downsample the LiDAR point cloud to this density. The density is in points per cubic meter.")
    args = parser.parse_args()
    
    images_dir, colmap_dir, chunks_dir = setup_dirs(
        args.images_dir,
        args.global_colmap_dir, args.chunks_dir,
        args.project_dir
    )

    if args.use_slurm:
        slurm_args = [
            "sbatch" 
        ]
    submitted_jobs = []

    colmap_exe = "colmap.bat" if platform.system() == "Windows" else "colmap"
    start_time = time.time()

    ## First create raw_chunks, each chunk has its own colmap.
    print(f"chunking colmap from {colmap_dir} to {args.chunks_dir}/raw_chunks")
    ss_make_chunk_args = [
            "python", f"preprocess/ss_make_chunk.py",
            "--project_dir", args.project_dir,
            "--base_dir", os.path.join(colmap_dir, "sparse", "0"),
            "--images_dir", f"{images_dir}",
            "--output_path", f"{chunks_dir}/chunks",
        ]
    if args.LiDAR_initialisation:
        ss_make_chunk_args.append("--LiDAR_initialisation")

    if args.LiDAR_downsample_density > 0:
        ss_make_chunk_args.extend(["--LiDAR_downsample_density", str(args.LiDAR_downsample_density)])
    try:
        subprocess.run(ss_make_chunk_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing image_undistorter: {e}")
        sys.exit(1)

    # create chunks.txt file that concatenates all chunks center.txt and extent.txt files
    try:
        subprocess.run([
            "python", "preprocess/concat_chunks_info.py",
            "--base_dir", os.path.join(chunks_dir, "chunks"),
            "--dest_dir", colmap_dir
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing concat_chunks_info.sh: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"chunks successfully prepared in {(end_time - start_time)/60.0} minutes.")

