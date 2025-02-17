import json
import numpy as np
import struct
import os
import laspy
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm

def parse_json(json_file):

    print("Reading recording details...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    recordings = data['RecordingProperties']
    image_info = {}
    
    # Extract relevant info from the json (ImageId, timestamp, GPS coordinates)
    for record in recordings:
        image_info[record['ImageId']] = {
            'timestamp': record['RecordingTimeGps'],
            'x': record['Y'],
            'y': record['X'],
        }
    
    return image_info

def sort_images_by_time(image_info):

    print("Sorting images...")

    sorted_images = sorted(image_info.items(), key=lambda x: datetime.fromisoformat(x[1]['timestamp'].replace("Z", "+00:00")))
    return sorted_images

# COLMAP text file format utilities
def write_cameras_txt(filename, cameras):
    with open(filename, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: {}\n'.format(len(cameras)))
        for camera_id, params in cameras.items():
            param_str = ' '.join(map(str, params['params']))
            f.write(f'{camera_id} {params["model"]} {params["width"]} {params["height"]} {param_str}\n')

def write_images_txt(filename, images):
    with open(filename, 'w') as f:
        f.write('# Image list with one line of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME, \n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write('# Number of images: {}\n'.format(len(images)))
        for image_id, params in images.items():
            qvec_str = ' '.join(map(str, params['qvec']))
            tvec_str = ' '.join(map(str, params['tvec']))
            f.write(f'{image_id} {qvec_str} {tvec_str} {params["camera_id"]} {params["name"]}\n\n')

def write_points3D_txt(output_filename, input_files_path):
    # Merge all .laz files into a single file
    merged_file = os.path.join(input_files_path, "merged.laz")
    if not os.path.exists(merged_file):
        os.system(f"pdal merge {input_files_path}/*.laz {merged_file}")
    
    # Read the merged LAZ file
    las = laspy.read(merged_file)
    print(f"Read {len(las.points)} points from {merged_file}")
    
    # Extract coordinates
    X = las.x
    Y = las.y
    Z = las.z
    
    # Compute spatial extents
    min_x, max_x = X.min(), X.max()
    min_y, max_y = Y.min(), Y.max()
    min_z, max_z = Z.min(), Z.max()
    
    # Determine cell size for 3D grid
    cell_size = 1.0
    print(f"Cell (cube) side length: {cell_size}")
    
    # Assign points to 3D cubes
    cell_x = ((X - min_x) // cell_size).astype(int)
    cell_y = ((Y - min_y) // cell_size).astype(int)
    cell_z = ((Z - min_z) // cell_size).astype(int)
    
    cells = np.column_stack((cell_x, cell_y, cell_z))
    
    # Compute density per cube and collect kept indices
    unique_cells, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    max_count = counts.max()
    target_density = max_count // 100
    
    print(f"Maximum points in a cube: {max_count}")
    print(f"Target density (max_count / 100): {target_density}")
    
    kept_indices = []
    # Process cubes with progress bar
    for cell_idx, cell_count in tqdm(enumerate(counts), total=len(counts), desc="Downsampling cubes"):
        pts_in_cube = np.where(inverse == cell_idx)[0]
        if cell_count > target_density:
            # Randomly sample points in dense cubes
            sampled = np.random.choice(pts_in_cube, target_density, replace=False)
            kept_indices.extend(sampled.tolist())
        else:
            # Keep all points in sparse cubes
            kept_indices.extend(pts_in_cube.tolist())
    
    print(f"Selected {len(kept_indices)} points for output.")
    
    # Check for color data
    has_colour = hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue")
    
    # Write directly to COLMAP format during single pass
    with open(output_filename, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        
        # Write points with progress bar
        for point_id, idx in tqdm(enumerate(kept_indices, 1), total=len(kept_indices), desc="Writing points"):
            # Get coordinates
            x = las.x[idx]
            y = las.y[idx]
            z = las.z[idx]
            
            # Get color values (if available)
            if has_colour:
                r = las.red[idx] // 256  # 16-bit to 8-bit conversion
                g = las.green[idx] // 256
                b = las.blue[idx] // 256
            else:
                r = g = b = 0
            
            # Write formatted line
            f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.000000\n")
    
    print(f"Successfully wrote {len(kept_indices)} points to {output_filename}")

def compute_intrinsics(cube_face_size):
    f = cube_face_size / 2  # Focal length assuming 90° FOV
    cx, cy = cube_face_size / 2, cube_face_size / 2
    return [f, cx, cy]

def compute_extrinsics(face, vehicle_direction, yaw):
    # The yaw in degrees is the sum of the yaw, the vehicle direction and the face direction
    yaw_degrees = yaw + vehicle_direction + {
        'f': 0,
        'r': 90,
        'b': 180,
        'l': 270,
        'f1': 0,
        'f2': 45,
        'r1': 90,
        'r2': 135,
        'b1': 180,
        'b2': 225,
        'l1': 270,
        'l2': 315,
        'u1': 90,
        'u2': 270
    }[face]
    
    # Convert pitch and yaw from degrees to radians
    pitch_radians = np.radians(90 +{
        'f': 0,
        'r': 0,
        'b': 0,
        'l': 0,
        'f1': 0,
        'f2': 0,
        'r1': 0,
        'r2': 0,
        'b1': 0,
        'b2': 0,
        'l1': 0,
        'l2': 0,
        'u1': -45,
        'u2': -45
    }[face])
    yaw_radians = np.radians(yaw_degrees)

    # Rotation matrix for pitch (around the x-axis)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_radians), -np.sin(pitch_radians)],
        [0, np.sin(pitch_radians),  np.cos(pitch_radians)]
    ])

    # Rotation matrix for yaw (around the z-axis)
    R_z = np.array([
        [np.cos(yaw_radians), -np.sin(yaw_radians), 0],
        [np.sin(yaw_radians),  np.cos(yaw_radians), 0],
        [0,                   0,                   1]
    ])

    # Combined rotation matrix: R = R_x * R_z
    R = np.dot(R_x, R_z)
    return R


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def rotation_matrix_to_quaternion(R): # DEPRECATED
    q = np.zeros(4)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s

    return q

def calculate_translation(rotation_matrix, position):
    """
    Calculate the translation vector given a rotation matrix and position.

    Args:
        rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.
        position (numpy.ndarray): A 1x3 or 3x1 vector (camera position in 3D space).

    Returns:
        numpy.ndarray: The translation vector (3x1).
    """
    # str(float(parts[5]) + 136500), str(float(parts[6]) + 455700)
    position[0] = float(position[0])
    position[1] = float(position[1])
    # Ensure the position is a column vector
    position = np.array(position).reshape(3, 1)
    
    # Compute the translation vector
    translation = -np.dot(rotation_matrix, position)
    
    return translation.flatten()

import subprocess

def convert_txt_to_bin(input_path, output_path):
    """
    Converts a COLMAP model from text format (.txt) to binary format (.bin).

    Args:
        input_path (str): Path to the folder containing the .txt model files.
        output_path (str): Path to the folder where the .bin model will be saved.

    Raises:
        RuntimeError: If the COLMAP command fails.
    """
    try:
        subprocess.run(
            [
                "colmap", "model_converter",
                "--input_path", input_path,
                "--output_path", output_path,
                "--output_type", "BIN"
            ],
            check=True
        )
        print(f"Successfully converted model from TXT to BIN. Output saved in: {output_path}")
        # Remove the input files directory
        os.system(f"rm -r {input_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"COLMAP model_converter failed: {e}")
    except FileNotFoundError:
        raise RuntimeError("COLMAP executable not found. Make sure COLMAP is installed and added to your PATH.")

def main(recording_details_path, output_dir, output_dir_bin, cube_face_size, input_laz_path, faces):

    start_time = datetime.now()

    os.makedirs(output_dir, exist_ok=True)

    # Load recording details
    with open(recording_details_path, 'r') as f:
        metadata = json.load(f)

    image_info = parse_json(recording_details_path)
    sorted_images = sort_images_by_time(image_info)  
    
    cameras = {
        1: {
            'model': 'SIMPLE_PINHOLE',  # Was SIMPLE_PINHOLE
            'width': cube_face_size,
            'height': cube_face_size,
            'params': compute_intrinsics(cube_face_size),
        }
    }

    images = {}
    image_id = 1

    for record in metadata['RecordingProperties']:
        x_rd, y_rd = record['X'], record['Y']
        height = record['Height']
        vehicle_direction = record['VehicleDirection']
        yaw = record['Yaw']

        for face in faces:

            # Extrinsic parameters
            rotation_matrix = compute_extrinsics(face, vehicle_direction, yaw)
            position = [x_rd, y_rd, height]  # Camera position (x, y, z)

            translation_vector = calculate_translation(rotation_matrix, position)

            # The images are stored in a folder called inputs. Inside there are 4 folders, one for each face of the cube, they are named cam<cam_n>
            # f = 1, r = 2, b = 3, l = 4
            # The images are named as follows: cam<cam_n>/<index>_<ImageId>_<face>.jpg where the index is a number given by the position in the sorted_images list
            # Find the index of the current image in the sorted_images list
            idx = next(i for i, (image_id, _) in enumerate(sorted_images) if image_id == record["ImageId"])
            idx = str(idx).zfill(4)
            cam_n = {
                'f': 1,
                'r': 2,
                'b': 3,
                'l': 4,
                'f1': 1,
                'f2': 2,
                'r1': 3,
                'r2': 4,
                'b1': 5,
                'b2': 6,
                'l1': 7,
                'l2': 8,
                'u1': 9,
                'u2': 10
            }[face]
            image_name = f"cam{cam_n}/{idx}_{record['ImageId']}_{face}.jpg"

            # Convert rotation matrix to quaternion
            qvec = rotmat2qvec(rotation_matrix)

            images[image_id] = {
                'qvec': qvec,
                'tvec': translation_vector,
                'camera_id': 1,  # Single camera ID for all images
                'name': image_name,
            }

            image_id += 1

    # Write COLMAP files
    write_cameras_txt(os.path.join(output_dir, 'cameras.txt'), cameras)
    write_images_txt(os.path.join(output_dir, 'images.txt'), images)
    write_points3D_txt(os.path.join(output_dir, 'points3D.txt'), input_laz_path)

    if not os.path.exists(output_dir_bin):
        os.makedirs(output_dir_bin)
    # Convert COLMAP files to binary format
    convert_txt_to_bin(output_dir, output_dir_bin)

    print(f"Total time taken: {datetime.now() - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=str, default="sfm",  help="Preprocessing workflow to execute. Options: sfm, cal_sfm, cal")

    args = parser.parse_args()
    # Example usage
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    recording_details_path = f"{base_path}/ss_raw_images/recording_details.json"
    output_dir = f"{base_path}/colmap_output"
    output_dir_bin = f"{base_path}/camera_calibration/unrectified/sparse/0"
    input_laz_path = f"{base_path}/ss_raw_images/LiDAR"
    cube_face_size = 1536
    choice = 0
    while choice not in ["1", "2", "3"]:
            print("Choose the directions you want to use:")
            print("1. Forward, Right, Backward, Left")
            print("2. Forward1, Forward2, Right1, Right2, Backward1, Backward2, Left1, Left2")
            print("3. Forward1, Forward2, Right1, Right2, Backward1, Backward2, Left1, Left2, Up1, Up2")
            choice = input("Enter your choice: ")
            # Define the directions \
            if choice == "1":
                faces = ['f', 'r', 'b', 'l']
            elif choice == "2":
                faces = ['f1', 'f2', 'r1', 'r2', 'b1', 'b2', 'l1', 'l2']
            elif choice == "3":
                faces = ['f1', 'f2', 'r1', 'r2', 'b1', 'b2', 'l1', 'l2', 'u1', 'u2']
            else:
                print("Invalid choice. Please try again.")
    main(recording_details_path, output_dir, output_dir_bin, cube_face_size, input_laz_path, faces)