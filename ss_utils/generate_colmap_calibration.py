'''
Thesis Project: Street-sparse-3DGS
Author: Iacopo Ermacora
Date: 11/2024-06/2025

Description:
This script generates COLMAP calibration files from a cyclomedia recording_details JSON file.
It processes the images, computes the camera intrinsics and extrinsics, and generates the 
necessary COLMAP files (cameras.bin, images.bin, points3D.bin) and the recording_details files
(for training and testing).
It is possible to run the script in evaluation mode but the recording_details.json file must come
from a 1mt spaced recording.
'''

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
from read_write_model import rotmat2qvec, write_cameras_binary, write_images_binary, write_points3D_binary, Camera, Image
import shutil

def parse_json(json_file):
    """
    Parse the recordin details JSON file to extract image information and metadata.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        tuple: A tuple containing:
            - image_info (dict): A dictionary mapping ImageId to its properties (timestamp, x, y).
            - metadata (dict): The entire JSON data from the RecordinProperties (list of images) loaded into a dictionary.
    """
    print("Reading recording details...")
    with open(json_file, 'r') as f:
        metadata = json.load(f)

    recordings = metadata['RecordingProperties']
    image_info = {}
    
    # Extract relevant info from the json (ImageId, timestamp, GPS coordinates)
    for record in recordings:
        image_info[record['ImageId']] = {
            'timestamp': record['RecordingTimeGps'],
            'x': record['Y'],
            'y': record['X'],
        }
    
    return image_info, metadata

def parse_iso_timestamp(timestamp_str):
    """
    Parse ISO format timestamp strings that may have variable precision in fractional seconds.
    
    Args:
        timestamp_str (str): ISO format timestamp string, e.g. "2023-10-23T10:30:32.24Z"
        
    Returns:
        datetime: Parsed datetime object
    """
    # Remove the 'Z' timezone indicator
    if timestamp_str.endswith('Z'):
        timestamp_str = timestamp_str[:-1]
    
    # Handle the case where we might have less than 6 digits for microseconds
    if '.' in timestamp_str:
        main_part, frac_part = timestamp_str.split('.')
        # Ensure the fractional part has exactly 6 digits (microseconds)
        frac_part = frac_part.ljust(6, '0')
        timestamp_str = f"{main_part}.{frac_part}"
    
    # Add UTC timezone info
    timestamp_str += "+00:00"
    
    return datetime.fromisoformat(timestamp_str)

def sort_images_by_time(image_info):
    """
    Sort images by their timestamp.
    
    Args:
        image_info (dict): A dictionary mapping ImageId to its properties (timestamp, x, y).
    
    Returns:
        list: A sorted list of tuples (ImageId, properties) based on timestamp.
    """
    print("Sorting images...")
    sorted_images = sorted(image_info.items(), key=lambda x: parse_iso_timestamp(x[1]['timestamp']))
    return sorted_images

def select_eval_images(sorted_images, metadata):
    """
    Select images for evaluation based on time and distance constraints:
    - Select one image every 4 in order of time
    - If one of the 4 images after the selected one is distant more than 2m from the previous one,
      reset the count and select that one
    - For COLMAP conversion, also include the second image after the selected one (third of every 5)
      if the distance constraint is met (this will be used for test.txt)
    
    Returns:
    - train_images: List of ImageIds that will be used for training
    - colmap_images: List of ImageIds for COLMAP conversion (includes both training and test images)
    - test_images: List of ImageIds for test.txt
    """
    print("Selecting images for evaluation...")
    train_images = []
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
        train_images.append(current_img_id)
        colmap_images.append(current_img_id)
        
        # Check the next 4 images for distance constraint
        reset_count = False
        next_selected_idx = i + 5  # Default next image to select (5 images ahead)
        
        # Check the distance to the next 4 images
        checking_x, checking_y = current_x, current_y
        for j in range(1, 5):
            if i + j >= len(sorted_images):
                break
            
            # Get the next image ID and its record
            next_img_id = sorted_images[i + j][0]
            next_record = image_to_record[next_img_id]
            next_x, next_y = next_record['X'], next_record['Y']
            
            # Calculate the euclidean distance
            distance = math.sqrt((next_x - checking_x)**2 + (next_y - checking_y)**2)
            
            # If the distance is greater than 2m (the images should technically be 1mt spaced), reset the count
            if distance > 2:
                reset_count = True
                next_selected_idx = i + j
                break
            else:
                checking_x, checking_y = next_x, next_y
        
        # If we're not resetting the count due to distance, 
        # add the third image (i+2) to test_images if it exists
        # but only add the images 35% of the time
        if not reset_count and i + 2 < len(sorted_images):
            if np.random.rand() < 0.35:
                # Add the third image to test_images
                test_img_id = sorted_images[i + 2][0]
                test_images.append(test_img_id)
                colmap_images.append(test_img_id)
        
        # Move to the next image to select
        i = next_selected_idx
    
    return train_images, colmap_images, test_images

def create_image_id_to_index_mapping(colmap_image_ids, metadata):
    """
    Create a consistent mapping from ImageId to index based on time-sorted colmap_image_ids. Necessary for consistent naming of 
    the images in the colmap files.
    
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
    sorted_pairs = sorted(id_time_pairs, key=lambda x: parse_iso_timestamp(x[1]))
    
    # Create the mapping
    image_id_to_index = {}
    for i, (img_id, _) in enumerate(sorted_pairs):
        image_id_to_index[img_id] = str(i).zfill(4)
    
    return image_id_to_index

def generate_test_file(test_images, metadata, faces, output_dir, image_id_to_index):
    """
    Generate test.txt file with image names using consistent indexes
    
    Args:
        test_images (list): List of ImageIds to include in the test.txt file.
        metadata (dict): The original metadata containing all images.
        faces (list): List of cube faces to include in the test.txt file.
        output_dir (str): Path to save the test.txt file.
        image_id_to_index (dict): Mapping from ImageId to zero-padded index string.
        
    Returns:
        None
    """
    print("Generating test.txt file...")
    
    # Map from ImageId to the full record
    metadata_records = {record['ImageId']: record for record in metadata['RecordingProperties']}
    
    # Sort test images by time for consistent ordering
    test_images_info = [(img_id, metadata_records[img_id]) for img_id in test_images]
    sorted_test_images = sorted(
        test_images_info, 
        key=lambda x: parse_iso_timestamp(x[1]['RecordingTimeGps'])
    )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create test.txt file
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for img_id, record in sorted_test_images:
            # Use the consistent index from the mapping
            idx_str = image_id_to_index[img_id]
            
            for face in faces:
                # Get camera number for the current face
                if args.directions == "1":
                    cam_n = {
                        'f1': 1,
                        'r1': 2,
                        'b1': 3,
                        'l1': 4
                    }[face]
                elif args.directions == "2" or args.directions == "3":
                    cam_n = {
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
                elif args.directions == "4":
                    cam_n = {
                        'f1': 1,
                        'r1': 2,
                        'b1': 3,
                        'l1': 4,
                        'u1': 5,
                        'u2': 6
                    }[face]
                image_name = f"cam{cam_n}/{idx_str}_{img_id}_{face}.jpg"
                f.write(f"{image_name}\n")
    
    print(f"Test file generated at {os.path.join(output_dir, 'test.txt')}")

def create_filtered_json(image_ids, metadata, output_path, filename):
    """
    Create a filtered JSON file with only the specified image IDs
    
    Args:
        image_ids (list): List of ImageIds to include in the new JSON file.
        metadata (dict): The original metadata containing all images.
        output_path (str): Path to save the new JSON file.
        filename (str): Name of the new JSON file.
    
    Returns:
        None
    """
    print(f"Creating {filename}...")
    
    # Create a new metadata object with only the selected records
    new_metadata = metadata.copy()
    new_metadata['RecordingProperties'] = [
        record for record in metadata['RecordingProperties'] 
        if record['ImageId'] in image_ids
    ]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Save the new metadata to the specified json file
    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(new_metadata, f, indent=4)
    
    print(f"{filename} generated at {os.path.join(output_path, filename)}")

def compute_intrinsics(cube_face_size):
    f = cube_face_size / 2  # Focal length assuming 90° FOV
    cx, cy = cube_face_size / 2, cube_face_size / 2
    return [f, cx, cy]

def compute_extrinsics(face, vehicle_direction, yaw):
    # The yaw in degrees is the sum of the yaw, the vehicle direction and the face direction
    yaw_degrees = yaw + vehicle_direction + {
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

def main(recording_details_path, output_dir_bin, cube_face_size, faces, eval_mode=False):
    
    start_time = datetime.now()

    # Create output directory if it doesn't exist
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

        # Select test images based on time and distance costraints
        train_images, colmap_image_ids, test_images = select_eval_images(sorted_images, metadata)
        
        # Create recording_details_train.json with the images selected for training
        create_filtered_json(train_images, metadata, os.path.join(args.project_dir, "camera_calibration", "extras"), 'recording_details_train.json')
        
        # Create recording_details_test.json with the images selected for both training and testing
        create_filtered_json(test_images, metadata, os.path.join(args.project_dir, "camera_calibration", "extras"), 'recording_details_test.json')
        
        # Create consistent mapping from ImageId to index
        image_id_to_index = create_image_id_to_index_mapping(colmap_image_ids, metadata)
        
        # Generate test.txt with consistent indexes
        generate_test_file(test_images, metadata, faces, os.path.join(args.project_dir, "camera_calibration", "extras"), image_id_to_index)
    else:
        print("Running in normal mode")
        # Get all image IDs
        all_image_ids = [img_id for img_id, _ in sorted_images]
        
        # Just copy recording_details.json to recording_details_train.json (there are not train/test images, all images are used for training)
        truth_path = os.path.join(os.path.join(args.project_dir, "camera_calibration", "extras"), 'recording_details_train.json')
        shutil.copy(recording_details_path, truth_path)
        print(f"Created {truth_path}")
        
        # Also copy recording_details.json to recording_details_test.json (there are not train/test images, all images are used for training)
        colmap_path = os.path.join(os.path.join(args.project_dir, "camera_calibration", "extras"), 'recording_details_test.json')
        shutil.copy(recording_details_path, colmap_path)
        print(f"Created {colmap_path}")
        
        # In normal mode, use all images for COLMAP
        colmap_image_ids = all_image_ids
        
        # Create consistent mapping from ImageId to index
        image_id_to_index = create_image_id_to_index_mapping(colmap_image_ids, metadata)
    
    # Compute the translation values to center the model at (0,0), as required by COLMAP
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
    1: Camera(id=1, # Add the camera ID here
              model='SIMPLE_PINHOLE',
              width=cube_face_size,
              height=cube_face_size,
              params=np.array(compute_intrinsics(cube_face_size))) # Ensure params is a numpy array if needed by the writer
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
            if args.directions == "1":
                cam_n = {
                    'f1': 1,
                    'r1': 2,
                    'b1': 3,
                    'l1': 4
                }[face]
            elif args.directions == "2" or args.directions == "3":
                cam_n = {
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
            elif args.directions == "4":
                cam_n = {
                    'f1': 1,
                    'r1': 2,
                    'b1': 3,
                    'l1': 4,
                    'u1': 5,
                    'u2': 6
                }[face]
            image_name = f"cam{cam_n}/{idx_str}_{record['ImageId']}_{face}.jpg"

            qvec = rotmat2qvec(rotation_matrix)
            images[image_id] = Image(id=image_id, # Use the current image_id loop variable
                         qvec=qvec,
                         tvec=translation_vector,
                         camera_id=1, # Or the relevant camera ID if dynamic
                         name=image_name,
                         xys=np.empty((0, 2)), # Empty placeholder for 2D points
                         point3D_ids=np.full(0, -1, dtype=np.int64)) # Empty placeholder for 3D point IDs
            image_id += 1
    
    # Create an empty points3D dictionary
    points3D = {}

    # Write binary files directly
    print(f"Writing binary COLMAP files to {output_dir_bin}...")
    write_cameras_binary(cameras, os.path.join(output_dir_bin, 'cameras.bin')) #
    write_images_binary(images, os.path.join(output_dir_bin, 'images.bin')) #
    write_points3D_binary(points3D, os.path.join(output_dir_bin, 'points3D.bin')) #
    print("Binary COLMAP files generated successfully.")

    end_time = datetime.now()
    print(f"Processing completed in {end_time - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True, help="Path to the project directory")
    parser.add_argument('--eval', action='store_true', help="Enable evaluation mode to select one image every 5 and check distance constraints")
    parser.add_argument('--directions', type=str, default='3', choices=['1', '2', '3', '4'], help="Choose the directions for the cube faces: 1 (f,r,b,l), 2 (f1,f2,r1,r2,b1,b2,l1,l2), 3 (f1,f2,r1,r2,b1,b2,l1,l2,u1,u2)")

    # Additional arguments to tweak certain parameters
    parser.add_argument('--cube_face_size', type=int, default=1536, help="Size of the cube face in pixels")
    args = parser.parse_args()
    
    # Define paths and parameters
    recording_details_path = f"{args.project_dir}/ss_raw_images/recording_details.json"
    output_dir_bin = f"{args.project_dir}/camera_calibration/unrectified/sparse/0/"
    cube_face_size = args.cube_face_size

    if args.directions == "1":
        faces = ['f1', 'r1', 'b1', 'l1']
    elif args.directions == "2":
        faces = ['f1', 'f2', 'r1', 'r2', 'b1', 'b2', 'l1', 'l2']
    elif args.directions == "3":
        faces = ['f1', 'f2', 'r1', 'r2', 'b1', 'b2', 'l1', 'l2', 'u1', 'u2']
    elif args.directions == "4":
        faces = ['f1', 'r1', 'b1', 'l1', 'u1', 'u2']
    main(recording_details_path, output_dir_bin, cube_face_size, faces, args.eval)