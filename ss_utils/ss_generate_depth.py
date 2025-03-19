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
    args = parser.parse_args()

    # Prompt the user to prevent execution of the script if not in a x server environment
    print("Please make sure you are running this script in an X server environment. If not this script will fail.")
    choice = input("Do you want to continue? [y/n]: ")
    if choice.lower() != 'y':
        print("Exiting the script.")
        sys.exit(0)

    chunk_dir = os.path.join(args.project_dir, "camera_calibration/chunks")
    
    total_ply = os.path.join(args.project_dir, "camera_calibration/depth_files/vis2mesh/total.ply")

    # Call the colmap_to_vis2mesh.py script to convert the COLMAP model to a custom JSON format

    print("#"*30)
    print("Step 1/6: Converting the COLMAP model to a custom JSON format for vis2mesh")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    # If the json file already exists, skip this step
    if not os.path.exists(os.path.join(args.project_dir, "camera_calibration", "depth_files", "vis2mesh", "total.ply_WORK", "cam0.json")):
        colmap_to_vis2mesh_args = [
                    "python", f"ss_utils/colmap_to_vis2mesh.py",
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
    print("Step 2/6: Generating the mesh using vis2mesh")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    # The vis2mesh directory should be relative to your current directory
    host_path = os.environ.get('HOST_PATH')
    mount_path = os.environ.get('MOUNT_PATH')

    # If the mesh already exists, skip this step
    if not os.path.exists(os.path.join(args.project_dir, "camera_calibration", "depth_files", "vis2mesh", "total_vis2mesh.ply")):
        docker_vis2mesh_command = [
        "sudo", "docker", "run", "-it",
        "-v", f"{host_path}/vis2mesh:/workspace",
        "-v", f"{mount_path}/camera_calibration/depth_files/vis2mesh/:/files",
        "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
        "--privileged", "--net", "host",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
        "-e", "DISPLAY=unix$DISPLAY",
        "--device=/dev/dri",
        "--gpus", "all",
        "vis2mesh", "/workspace/inference.py", "/files/total.ply", "--cam", "cam0"
        ]
        try:
            subprocess.run(docker_vis2mesh_command, check=True, text=True,)
        except subprocess.CalledProcessError as e:
            print(f"Error executing docker_vis2mesh_command: {e}")
            sys.exit(1)
    
        print(f"Time taken to generate the mesh using vis2mesh: {time.time() - start_time} seconds")
    else:
        print("The mesh already exists. Skipping this step")

    # Call the script to convert the mesh to the .ctm format

    print("#"*30)
    print("Step 3/6: Converting the mesh to the .ctm format")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    # If the ctm folder exists, skip this step
    if not os.path.exists(os.path.join(args.project_dir, "camera_calibration", "depth_files", "ctm_meshes")):
        ply_mesh_to_ctm = [
                    "python", f"ss_utils/ply_mesh_to_ctm.py",
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
    print("Step 4/6: Converting the recording_details file to the .stations format")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    # If the stations file already exists, skip this step
    if not os.path.exists(os.path.join(args.project_dir, "camera_calibration", "depth_files", "total.stations")):
        json_rds_to_stations = [
                    "/cityfusion_master/build/debug/experimental/cityfusion/tools/ml/gaussian_splatting/json_rds_to_stations/json_rds_to_stations",
                    f"{args.project_dir}/ss_raw_images/recording_details.json",
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
    print("Step 5/6: Rendering the depth maps")
    print("#"*30)

    # Get the time before the script starts
    start_time = time.time()

    if not os.path.exists(os.path.join(args.project_dir, "camera_calibration/depth_files/rgb_depths")):
        os.makedirs(os.path.join(args.project_dir, "camera_calibration/depth_files/rgb_depths"))

        render_depth_gaussians = [
            '/cityfusion_master/build/debug/experimental/cityfusion/tools/ml/gaussian_splatting/render_depth_gaussians/render_depth_gaussians',
            '--in_cell_prefix="pmg__hires_"',
            f'--in_terrestrial_stations_file_url=file:///{args.project_dir}/camera_calibration/depth_files/total.stations',
            f'--in_mesh_directory=file:///{args.project_dir}/camera_calibration/depth_files/ctm_meshes',
            f'--out_depth_cyclo_directory_url=file:///{args.project_dir}/camera_calibration/depth_files/rgb_depths',
            '--10images'
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
    print("Step 6/6: Converting the depth maps to the bw format and sorting them")
    print("#"*30)

    # depth_map_to_distances = [
    #     "python", f"ss_utils/depth_map_to_distances.py",
    #     "--input_dir", os.path.join(args.project_dir, f"{args.project_dir}/camera_calibration/depth_files/rgb_depths"),
    #     "--output_dir", os.path.join(args.project_dir, ), # TODO: Add the path to the output directory
    #     "--chunks_dir", os.path.join(args.project_dir, ), # TODO: Add the path to the chunks directory
    # ]

    # try:
    #     subprocess.run(depth_map_to_distances, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error executing depth_map_to_distances: {e}")
    #     sys.exit(1)
