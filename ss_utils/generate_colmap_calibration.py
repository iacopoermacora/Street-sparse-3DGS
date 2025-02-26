import json
import numpy as np
import struct
import os
import laspy
from datetime import datetime
import argparse
from tqdm import tqdm
import math

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

def write_added_images_txt(filename, images):
    with open(filename, 'w') as f:
        f.write('# Image list with one line of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME, \n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write('# Number of images: {}\n'.format(len(images)))
        for image_id, params in images.items():
            qvec_str = ' '.join(map(str, params['qvec']))
            tvec_str = ' '.join(map(str, params['tvec']))
            f.write(f'{image_id} {qvec_str} {tvec_str} {params["camera_id"]} {params["name"]}\n\n')

def write_points3D_txt(output_filename):
    
    # Write directly to COLMAP format during single pass
    with open(output_filename, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')

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

def main(recording_details_path, output_dir, output_dir_bin, output_dir_added, output_dir_bin_added, cube_face_size, faces):

    start_time = datetime.now()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_added, exist_ok=True)

    # Load recording details
    with open(recording_details_path, 'r') as f:
        metadata = json.load(f)

    # Create a mapping from ImageId to the full record from metadata['RecordingProperties']
    metadata_records = { record['ImageId']: record for record in metadata['RecordingProperties'] }

    image_info = parse_json(recording_details_path)
    # Assume sort_images_by_time returns a list of tuples: (ImageId, record)
    sorted_images = sort_images_by_time(image_info)  

    cameras = {
        1: {
            'model': 'SIMPLE_PINHOLE',
            'width': cube_face_size,
            'height': cube_face_size,
            'params': compute_intrinsics(cube_face_size),
        }
    }

    images = {}
    image_id = 1

    # Process original images from the recording file.
    for record in metadata['RecordingProperties']:
        x_rd, y_rd = record['X'], record['Y']
        height = record['Height']
        vehicle_direction = record['VehicleDirection']
        yaw = record['Yaw']

        # Use the sorted order to get an index for naming.
        idx = next(i for i, (img_id, _) in enumerate(sorted_images) if img_id == record["ImageId"])
        idx_str = str(idx).zfill(4)

        for face in faces:
            # Compute extrinsics
            rotation_matrix = compute_extrinsics(face, vehicle_direction, yaw)
            position = [x_rd - 136757, y_rd - 455750, height] # xi - 136757, yi - 455750
            translation_vector = calculate_translation(rotation_matrix, position)

            # Get camera number for the current face.
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
            image_name = f"cam{cam_n}/{idx_str}_{record['ImageId']}_{face}.jpg"

            qvec = rotmat2qvec(rotation_matrix)
            images[image_id] = {
                'qvec': qvec,
                'tvec': translation_vector,
                'camera_id': 1,
                'name': image_name,
            }
            image_id += 1

    # ---------------------------------------------------------------------------
    # Add intermediate captures if the distance between consecutive captures 
    # is approximately 5 meters (with a tolerance of ±0.5 m).
    # ---------------------------------------------------------------------------
    added_images = images.copy()
    added_image_id = image_id

    # Iterate over consecutive pairs from the sorted images.
    # Here we assume each element in sorted_images is (ImageId, record)
    for i in range(len(sorted_images) - 1):
        img_id1, _ = sorted_images[i]
        img_id2, _ = sorted_images[i+1]

        # Retrieve full records with uppercase keys
        record1 = metadata_records[img_id1]
        record2 = metadata_records[img_id2]

        # Calculate 2D distance (using X and Y; adjust if you want to include height)
        dx = record2['X'] - record1['X']
        dy = record2['Y'] - record1['Y']
        distance = math.sqrt(dx**2 + dy**2)

        # If the distance is roughly 5 m, add an interpolated capture.
        if 3 <= distance <= 10:
            # Interpolate properties between record1 and record2.
            new_record = {}
            for field in ['X', 'Y', 'Height', 'VehicleDirection', 'Yaw']:
                new_record[field] = (record1[field] + record2[field]) / 2

            # Interpolate the recording time if present.
            time1 = datetime.fromisoformat(record1['RecordingTimeGps'])
            time2 = datetime.fromisoformat(record2['RecordingTimeGps'])
            new_time = time1 + (time2 - time1) / 2
            new_record['RecordingTimeGps'] = new_time.isoformat()

            # Create a new unique ImageId for the added capture.
            new_record['ImageId'] = f"added_{added_image_id}"
            idx_added = str(added_image_id).zfill(4)

            # For each face, compute and add the image.
            for face in faces:
                rotation_matrix = compute_extrinsics(face, new_record['VehicleDirection'], new_record['Yaw'])
                position = [new_record['X'] - 136757, new_record['Y'] - 455750, new_record['Height']] # xi - 136757, yi - 455750
                translation_vector = calculate_translation(rotation_matrix, position)

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
                image_name = f"cam{cam_n}/{idx_added}_{new_record['ImageId']}_{face}.jpg"
                qvec = rotmat2qvec(rotation_matrix)
                added_images[added_image_id] = {
                    'qvec': qvec,
                    'tvec': translation_vector,
                    'camera_id': 1,
                    'name': image_name,
                }
                added_image_id += 1

    # Write COLMAP files for the original captures.
    write_cameras_txt(os.path.join(output_dir, 'cameras.txt'), cameras)
    write_cameras_txt(os.path.join(output_dir_added, 'cameras.txt'), cameras)

    write_points3D_txt(os.path.join(output_dir, 'points3D.txt'))
    write_points3D_txt(os.path.join(output_dir_added, 'points3D.txt'))

    write_images_txt(os.path.join(output_dir, 'images.txt'), images)
    # Write the added (interpolated) images to a separate file.
    write_images_txt(os.path.join(output_dir_added, 'images.txt'), added_images)

    if not os.path.exists(output_dir_bin):
        os.makedirs(output_dir_bin)
    # Convert COLMAP files to binary format
    convert_txt_to_bin(output_dir, output_dir_bin)

    if not os.path.exists(output_dir_bin_added):
        os.makedirs(output_dir_bin_added)
    # Convert COLMAP files to binary format
    convert_txt_to_bin(output_dir_added, output_dir_bin_added)

    end_time = datetime.now()
    print(f"Processing completed in {end_time - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=str, default="sfm",  help="Preprocessing workflow to execute. Options: sfm, cal_sfm, cal")

    args = parser.parse_args()
    # Example usage
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    recording_details_path = f"{base_path}/ss_raw_images/recording_details.json"
    output_dir = f"{base_path}/colmap_output"
    output_dir_added = f"{base_path}/colmap_output_added"
    output_dir_bin = f"{base_path}/camera_calibration/unrectified/sparse/0"
    output_dir_bin_added = f"{base_path}/colmap_output_added_bin"
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
    main(recording_details_path, output_dir, output_dir_bin, output_dir_added, output_dir_bin_added, cube_face_size, faces)