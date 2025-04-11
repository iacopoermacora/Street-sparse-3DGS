'''
Thesis Project: Street-sparse-3DGS
Author: Iacopo Ermacora
Date: 11/2024-06/2025

Description: 
This script generates depth maps from a COLMAP model using multiple steps.
Vis2mesh is used to generate the mesh from the COLMAP model.
The rest of the pipeline is used to convert the mesh to a custom format and render depth maps following the cyclomedia pipeline. 
'''

import subprocess
import os
import sys
import argparse
import open3d as o3d
import numpy as np
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True)
    parser.add_argument('--directions', type=str, default='3', choices=['1', '2', '3', '4'], help='Camera directions: 1=FRLB, 2=F1F2R1R2B1B2L1L2, 3=F1F2R1R2B1B2L1L2U1U2')
    args = parser.parse_args()

    # Prompt the user to prevent execution of the script if not in a x server environment
    print("Please make sure you are running this script in an X server environment. If not this script will fail.")
    choice = input("Do you want to continue? [y/n]: ")
    if choice.lower() != 'y':
        print("Exiting the script.")
        sys.exit(0)

    chunk_dir = os.path.join(args.project_dir, "camera_calibration/chunks")
    
    total_ply = os.path.join(args.project_dir, "camera_calibration/depth_files/vis2mesh/total.ply")

    # Call the script to augment the recording details

    print("#"*30)
    print("Step 1/7: Augmenting the recording details")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    # If the augmented recording details file already exists, skip this step
    if not os.path.exists(os.path.join(args.project_dir, "ss_raw_images", "recording_details_augmented.json")):
        augment_recording_details = [
                    "python", f"ss_utils/depth_scripts/augment_recording_details.py",
                    "--project_dir", args.project_dir,
                    "--directions", args.directions
                ]
        try:
            subprocess.run(augment_recording_details, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing augment_recording_details: {e}")
            sys.exit(1)
        
        print(f"Time taken to augment the recording details: {time.time() - start_time} seconds")

    # Call the colmap_to_vis2mesh.py script to convert the COLMAP model to a custom JSON format

    print("#"*30)
    print("Step 2/7: Converting the COLMAP model to a custom JSON format for vis2mesh")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    # If the json file already exists, skip this step
    if not os.path.exists(os.path.join(args.project_dir, "camera_calibration", "depth_files", "vis2mesh", "total.ply_WORK", "cam0.json")):
        colmap_to_vis2mesh_args = [
                    "python", f"ss_utils/depth_scripts/colmap_to_vis2mesh.py",
                    "--model_dir", os.path.join(args.project_dir, "camera_calibration", "aligned", "sparse", "0"),
                    "--output_file", os.path.join(args.project_dir, "camera_calibration", "depth_files", "vis2mesh", "total.ply_WORK", "cam0.json"),
                ]
        try:
            subprocess.run(colmap_to_vis2mesh_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing colmap_to_vis2mesh: {e}")
            sys.exit(1)
        
        print(f"Time taken to convert the COLMAP model to a custom JSON format: {time.time() - start_time} seconds")
    else:
        print("The JSON file already exists. Skipping this step.")

    # Call the docker container vis2mesh to generate the mesh

    print("#"*30)
    print("Step 3/7: Generating the mesh using vis2mesh")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    # The vis2mesh directory should be relative to your current directory
    host_path = os.environ.get('HOST_PATH')
    mount_path = os.environ.get('MOUNT_PATH')
    home_path = os.environ.get('HOME_PATH')
    display_path = os.environ.get('DISPLAY')
    device_path = os.environ.get('DEVICE_PATH')

    # If the mesh already exists, skip this step
    if not os.path.exists(os.path.join(args.project_dir, "camera_calibration", "depth_files", "vis2mesh", "total_vis2mesh.ply")):

        print("Generating the mesh using vis2mesh will create no output in the terminal. Please wait for the process to complete.")
        print("You can check inside the vis2mesh folder for the generated mesh or for progress. This might take a while, in the order of hours possibly.")
        docker_vis2mesh_command = f"sudo docker run -i -v {host_path}/vis2mesh:/workspace \
            -v {mount_path}/camera_calibration/depth_files/vis2mesh/:/files -v \
            {home_path}/.Xauthority:/root/.Xauthority \
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
            --privileged --net host \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
            -e DISPLAY=unix{display_path} --device=/dev/dri \
            --gpus all vis2mesh \
            /workspace/inference.py /files/total.ply --cam cam0"
        
        try:
            # Start the process
            process = subprocess.Popen(
                docker_vis2mesh_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Read and print stdout in real-time
            for line in process.stdout:
                print(line, end='')  # Print without adding extra newlines
            
            # Wait for the process to complete
            process.wait()
            
            # Check for errors
            if process.returncode != 0:
                print(f"Process exited with return code {process.returncode}")
                for line in process.stderr:
                    print(f"Error: {line}", end='')
                    
        except Exception as e:
            print(f"Error executing docker command: {e}")
            sys.exit(1)
    
        print(f"Time taken to generate the mesh using vis2mesh: {time.time() - start_time} seconds")
    else:
        print("The mesh already exists. Skipping this step")

    # Call the script to convert the mesh to the .ctm format

    print("#"*30)
    print("Step 4/7: Converting the mesh to the .ctm format")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    # If the ctm folder exists, skip this step
    if not os.path.exists(os.path.join(args.project_dir, "camera_calibration", "depth_files", "ctm_meshes")):
        ply_mesh_to_ctm = [
                    "python", f"ss_utils/depth_scripts/ply_mesh_to_ctm.py",
                    "--project_dir", args.project_dir,
                    "--input_ply", os.path.join(args.project_dir, "camera_calibration", "depth_files", "vis2mesh", "total_vis2mesh.ply"),
                    "--output_dir", os.path.join(args.project_dir, "camera_calibration", "depth_files", "ctm_meshes"),
                ]
        try:
            subprocess.run(ply_mesh_to_ctm, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing ply_mesh_to_ctm: {e}")
            sys.exit(1)
        
        print(f"Time taken to convert the mesh to the .ctm format: {time.time() - start_time} seconds")
    else:
        print("The ctm folder already exists. Skipping this step")

    # Call the script to convert the recording_details file to the .stations format
    print("#"*30)
    print("Step 5/7: Converting the recording_details file to the .stations format")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    cityfusion_master_path = os.environ.get('CITYFUSION_MASTER_PATH')

    # Change the permissions of the depth_files folder
    os.chmod(os.path.join(args.project_dir, "camera_calibration", "depth_files"), 0o777)

    # If the stations file already exists, skip this step
    if not os.path.exists(os.path.join(args.project_dir, "camera_calibration", "depth_files", "total.stations")):
        json_rds_to_stations = [
                    "sudo", "docker", "run", "--rm",
                    "--gpus", "all",
                    "--net=host",
                    "-v", f"{mount_path}:/{args.project_dir}",
                    "-v", f"{cityfusion_master_path}:/cityfusion_master",
                    "-v", f"/media/raid/cityfusion_third_party:/media/raid/cityfusion_third_party",
                    "-e", f"LD_LIBRARY_PATH=/cityfusion_master/build/third_party/opt/cityfusion/lib",
                    "gpuproc02.cyclomedia001.cyclomedia.intern:5000/cm-gpuproc-dev:4.31",
                    "/cityfusion_master/build/debug/experimental/cityfusion/tools/gaussian_splatting/json_rds_to_stations/json_rds_to_stations",
                    f"{args.project_dir}/ss_raw_images/recording_details_augmented.json",
                    f"{args.project_dir}/camera_calibration/depth_files/total.stations",
                ]
        try:
            subprocess.run(json_rds_to_stations, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing json_rds_to_stations: {e}")
            sys.exit(1)
        
        print(f"Time taken to convert the recording_details file to the .stations format: {time.time() - start_time} seconds")

    else:
        print("The stations file already exists. Skipping this step")

    # Call the script to render the depth maps

    print("#"*30)
    print("Step 6/7: Rendering the depth maps")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    if os.path.exists(os.path.join(args.project_dir, "camera_calibration", "depth_files", "total.stations")) and not os.path.exists(os.path.join(args.project_dir, "camera_calibration/depth_files/rgb_depths")):
        os.makedirs(os.path.join(args.project_dir, "camera_calibration/depth_files/rgb_depths"))
        os.chmod(os.path.join(args.project_dir, "camera_calibration/depth_files/rgb_depths"), 0o777)

        render_depth_gaussians = [
            "sudo", "docker", "run", "--rm",
            "--gpus", "all",
            "--net=host",
            "-v", f"{mount_path}:/depth_data",
            "-v", f"{cityfusion_master_path}:/cityfusion_master",
            "-v", f"/media/raid/cityfusion_third_party:/media/raid/cityfusion_third_party",
            "-e", f"LD_LIBRARY_PATH=/cityfusion_master/build/third_party/opt/cityfusion/lib",
            "gpuproc02.cyclomedia001.cyclomedia.intern:5000/cm-gpuproc-dev:4.31",
            "/cityfusion_master/build/debug/experimental/cityfusion/tools/gaussian_splatting/render_depth_gaussians/render_depth_gaussians",
            "--in_cell_prefix=pmg__hires_",
            f"--in_terrestrial_stations_file_url=file:///depth_data/camera_calibration/depth_files/total.stations",
            f"--in_mesh_directory=file:///depth_data/camera_calibration/depth_files/ctm_meshes",
            f"--out_depth_cyclo_directory_url=file:///depth_data/camera_calibration/depth_files/rgb_depths",
            "--10images"
        ]
        try:
            subprocess.run(render_depth_gaussians, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing json_rds_to_stations: {e}")
            sys.exit(1)
        
        print(f"Time taken to render the depth maps: {time.time() - start_time} seconds")
    else:
        print("The depth maps already exist. Skipping this step")

    # Call the script to convert the depth maps to the bw format and sort them accordingly

    print("#"*30)
    print("Step 7/7: Converting the depth maps to the bw format and sorting them")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    # If the depth folder already exists, skip this step
    if not os.path.exists(os.path.join(args.project_dir, "camera_calibration/rectified/depth")):
        depth_map_to_distances = [
            "python", f"ss_utils/depth_scripts/depth_map_to_distances.py",
            "--project_dir", args.project_dir
        ]

        try:
            subprocess.run(depth_map_to_distances, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing depth_map_to_distances: {e}")
            sys.exit(1)
        
        print(f"Time taken to convert the depth maps to the bw format and sort them: {time.time() - start_time} seconds")
    else:
        print("The depth folder already exists. Skipping this step")
