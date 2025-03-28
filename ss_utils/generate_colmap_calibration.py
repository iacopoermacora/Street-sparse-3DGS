import json
import numpy as np
import struct
import os
import laspy
from datetime import datetime
import argparse
from tqdm import tqdm
import math
import subprocess
from read_write_model import rotmat2qvec

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
    
    return image_info, data

def sort_images_by_time(image_info):
    print("Sorting images...")
    sorted_images = sorted(image_info.items(), key=lambda x: datetime.fromisoformat(x[1]['timestamp'].replace("Z", "+00:00")))
    return sorted_images

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def select_eval_images(sorted_images, metadata, faces):
    """
    Select images for evaluation based on time and distance constraints:
    - Select one image every 5 in order of time
    - If one of the 4 images after the selected one is distant more than 2m from the previous one,
      reset the count and select that one
    - For COLMAP conversion, also include the second image after the selected one (third of every 5)
      if the distance constraint is met
    
    Returns:
    - selected_images: List of ImageIds for recording_details_truth.json
    - colmap_images: List of ImageIds for COLMAP conversion
    - test_images: List of ImageIds for test.txt (third images)
    """
    print("Selecting images for evaluation...")
    selected_images = []
    colmap_images = []
    test_images = []
    
    # Map from ImageId to record
    image_to_record = {record['ImageId']: record for record in metadata['RecordingProperties']}
    
    i = 0
    while i < len(sorted_images):
        current_img_id = sorted_images[i][0]
        current_record = image_to_record[current_img_id]
        current_x, current_y = current_record['X'], current_record['Y']
        
        # Select this image
        selected_images.append(current_img_id)
        colmap_images.append(current_img_id)
        
        # Check the next 4 images for distance constraint
        reset_count = False
        next_selected_idx = i + 5  # Default next image to select (5 images ahead)
        
        checking_x, checking_y = current_x, current_y
        for j in range(1, 5):
            if i + j >= len(sorted_images):
                break
                
            next_img_id = sorted_images[i + j][0]
            next_record = image_to_record[next_img_id]
            next_x, next_y = next_record['X'], next_record['Y']
            
            distance = calculate_distance(checking_x, checking_y, next_x, next_y)
            
            if distance > 2:  # If distance > 2 meters
                reset_count = True
                next_selected_idx = i + j
                break
            else:
                checking_x, checking_y = next_x, next_y
        
        # If we're not resetting the count due to distance, 
        # add the third image (i+2) to test_images if it exists
        if not reset_count and i + 2 < len(sorted_images):
            test_img_id = sorted_images[i + 2][0]
            test_images.append(test_img_id)
            colmap_images.append(test_img_id)
        
        # Move to the next image to select
        i = next_selected_idx
    
    return selected_images, colmap_images, test_images

def create_image_id_to_index_mapping(colmap_image_ids, metadata):
    """
    Create a consistent mapping from ImageId to index based on time-sorted colmap_image_ids.
    
    Args:
        colmap_image_ids: List of ImageIds to be included in COLMAP
        metadata: The recording metadata
        
    Returns:
        dict: Mapping from ImageId to zero-padded index string
    """
    print("Creating consistent image ID to index mapping...")
    
    # Create a list of (ImageId, timestamp) pairs
    id_time_pairs = []
    for img_id in colmap_image_ids:
        # Find the record for this ImageId
        for record in metadata['RecordingProperties']:
            if record['ImageId'] == img_id:
                timestamp = record['RecordingTimeGps']
                id_time_pairs.append((img_id, timestamp))
                break
    
    # Sort by timestamp
    sorted_pairs = sorted(id_time_pairs, key=lambda x: datetime.fromisoformat(x[1].replace("Z", "+00:00")))
    
    # Create the mapping
    image_id_to_index = {}
    for i, (img_id, _) in enumerate(sorted_pairs):
        image_id_to_index[img_id] = str(i).zfill(4)
    
    return image_id_to_index

def generate_test_file(test_images, metadata, faces, output_dir, image_id_to_index):
    """Generate test.txt file with image names using consistent indexes"""
    print("Generating test.txt file...")
    
    # Map from ImageId to the full record
    metadata_records = {record['ImageId']: record for record in metadata['RecordingProperties']}
    
    # Sort test images by time for consistent ordering
    test_images_info = [(img_id, metadata_records[img_id]) for img_id in test_images]
    sorted_test_images = sorted(
        test_images_info, 
        key=lambda x: datetime.fromisoformat(x[1]['RecordingTimeGps'].replace("Z", "+00:00"))
    )
    
    # Create test.txt file
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for img_id, record in sorted_test_images:
            # Use the consistent index from the mapping
            idx_str = image_id_to_index[img_id]
            
            for face in faces:
                # Get camera number for the current face
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
                image_name = f"cam{cam_n}/{idx_str}_{img_id}_{face}.jpg"
                f.write(f"{image_name}\n")
    
    print(f"Test file generated at {os.path.join(output_dir, 'test.txt')}")

def create_filtered_json(image_ids, metadata, output_path, filename):
    """Create a filtered JSON file with only the specified image IDs"""
    print(f"Creating {filename}...")
    
    # Create a new metadata object with only the selected records
    new_metadata = metadata.copy()
    new_metadata['RecordingProperties'] = [
        record for record in metadata['RecordingProperties'] 
        if record['ImageId'] in image_ids
    ]
    
    # Save the new metadata to the specified json file
    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(new_metadata, f, indent=4)
    
    print(f"{filename} generated at {os.path.join(output_path, filename)}")

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

def calculate_translation(rotation_matrix, position):
    """
    Calculate the translation vector given a rotation matrix and position.

    Args:
        rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.
        position (numpy.ndarray): A 1x3 or 3x1 vector (camera position in 3D space).

    Returns:
        numpy.ndarray: The translation vector (3x1).
    """
    position[0] = float(position[0])
    position[1] = float(position[1])
    # Ensure the position is a column vector
    position = np.array(position).reshape(3, 1)
    
    # Compute the translation vector
    translation = -np.dot(rotation_matrix, position)
    
    return translation.flatten()

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

def compute_centering_translation(metadata):
    """
    Compute translation values to center the model at (0,0) for x and y.
    
    Args:
        metadata (dict): The metadata containing RecordingProperties.
        
    Returns:
        tuple: (x_translation, y_translation) values to center the model.
    """
    # Extract all X and Y coordinates
    x_coords = [record['X'] for record in metadata['RecordingProperties']]
    y_coords = [record['Y'] for record in metadata['RecordingProperties']]
    
    # Calculate the center of the model
    x_center = sum(x_coords) / len(x_coords)
    y_center = sum(y_coords) / len(y_coords)
    
    return x_center, y_center

def main(recording_details_path, output_dir, output_dir_bin, cube_face_size, faces, eval_mode=False):
    import subprocess

    start_time = datetime.now()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_dir_bin), exist_ok=True)

    # Load recording details
    image_info, metadata = parse_json(recording_details_path)
    
    # Sort images by time
    sorted_images = sort_images_by_time(image_info)
    
    # Create a list of image IDs to include in COLMAP
    colmap_image_ids = []
    
    # Handle --eval parameter
    if eval_mode:
        print("Running in evaluation mode")
        # Select images based on time and distance
        selected_images, colmap_image_ids, test_images = select_eval_images(sorted_images, metadata, faces)
        
        # Create recording_details_train.json
        create_filtered_json(selected_images, metadata, os.path.dirname(recording_details_path), 'recording_details_train.json')
        
        # Create recording_details_train_test.json
        create_filtered_json(colmap_image_ids, metadata, os.path.dirname(recording_details_path), 'recording_details_train_test.json')
        
        # Create consistent mapping from ImageId to index
        image_id_to_index = create_image_id_to_index_mapping(colmap_image_ids, metadata)
        
        # Generate test.txt with consistent indexes
        generate_test_file(test_images, metadata, faces, os.path.dirname(recording_details_path), image_id_to_index)
    else:
        print("Running in normal mode")
        # Get all image IDs
        all_image_ids = [img_id for img_id, _ in sorted_images]
        
        # Just copy recording_details.json to recording_details_train.json
        import shutil
        truth_path = os.path.join(os.path.dirname(recording_details_path), 'recording_details_train.json')
        shutil.copy(recording_details_path, truth_path)
        print(f"Created {truth_path}")
        
        # Also copy recording_details.json to recording_details_train_test.json
        colmap_path = os.path.join(os.path.dirname(recording_details_path), 'recording_details_train_test.json')
        shutil.copy(recording_details_path, colmap_path)
        print(f"Created {colmap_path}")
        
        # In normal mode, use all images for COLMAP
        colmap_image_ids = all_image_ids
        
        # Create consistent mapping from ImageId to index
        image_id_to_index = create_image_id_to_index_mapping(colmap_image_ids, metadata)
    
    # Compute the translation values to center the model at (0,0)
    x_center, y_center = compute_centering_translation(metadata)
    
    # Save the translation values to translation.json
    translation_json = {
        "x_translation": float(x_center),
        "y_translation": float(y_center)
    }
    
    with open(os.path.join(args.project_dir, 'camera_calibration/translation.json'), 'w') as f:
        json.dump(translation_json, f, indent=4)
    
    print(f"Translation values saved to {os.path.join(args.project_dir, 'camera_calibration/translation.json')}")
    
    # Mapping from ImageId to the full record
    metadata_records = {record['ImageId']: record for record in metadata['RecordingProperties']}
    
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

    # Process filtered images for COLMAP
    if eval_mode:
        # When in eval mode, only process the selected images for COLMAP
        records_to_process = [metadata_records[img_id] for img_id in colmap_image_ids]
    else:
        # In normal mode, process all images
        records_to_process = metadata['RecordingProperties']

    # Process images for COLMAP
    for record in records_to_process:
        x_rd, y_rd = record['X'], record['Y']
        height = record['Height']
        vehicle_direction = record['VehicleDirection']
        yaw = record['Yaw']
        
        # Use the consistent index mapping
        idx_str = image_id_to_index[record['ImageId']]

        for face in faces:
            # Compute extrinsics
            rotation_matrix = compute_extrinsics(face, vehicle_direction, yaw)
            # Apply the calculated translation to center the model
            position = [x_rd - x_center, y_rd - y_center, height]
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

    # Write COLMAP files
    write_cameras_txt(os.path.join(output_dir, 'cameras.txt'), cameras)
    write_points3D_txt(os.path.join(output_dir, 'points3D.txt'))
    write_images_txt(os.path.join(output_dir, 'images.txt'), images)

    # Create the directory structure for the binary output
    os.makedirs(os.path.dirname(output_dir_bin), exist_ok=True)
    if not os.path.exists(os.path.dirname(output_dir_bin)):
        os.makedirs(os.path.dirname(output_dir_bin))
    
    # Convert COLMAP files to binary format
    convert_txt_to_bin(output_dir, output_dir_bin)

    end_time = datetime.now()
    print(f"Processing completed in {end_time - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=str, default="sfm",  help="Preprocessing workflow to execute. Options: sfm, cal_sfm, cal")
    parser.add_argument('--project_dir', type=str, required=True, help="Path to the project directory")
    parser.add_argument('--eval', action='store_true', help="Enable evaluation mode to select one image every 5 and check distance constraints")

    args = parser.parse_args()
    # Example usage
    recording_details_path = f"{args.project_dir}/ss_raw_images/recording_details.json"
    output_dir = f"{args.project_dir}/colmap_output"
    output_dir_bin = f"{args.project_dir}/camera_calibration/unrectified/sparse/0/"
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
    main(recording_details_path, output_dir, output_dir_bin, cube_face_size, faces, args.eval)