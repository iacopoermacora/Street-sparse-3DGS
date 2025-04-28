'''
Thesis Project: Street-sparse-3DGS
Author: Iacopo Ermacora
Date: 11/2024-06/2025

Description: This script augments the recording details of a dataset by interpolating 
between existing recordings. It generates new recordings based on the distance between
existing ones, and creates new COLMAP images for augmented data. The script also handles
the reading and writing of COLMAP model files, including images and their parameters.
'''

import json
import math
import random
import string
from datetime import datetime
import os
import numpy as np
import argparse
from tqdm import tqdm
import sys
import shutil
from read_write_model import Image, write_images_binary, read_model, rotmat2qvec

def calculate_distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points.
    Args:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.
    Returns:
        The Euclidean distance between the two points.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def generate_unique_image_name(all_image_names):
    """Generate a random 8-character alphanumeric ImageId that doesn't exist in all_image_names.
    
    Args:
        all_image_names (set): Set of existing image names to avoid duplicates.
    
    Returns:
        str: A unique 8-character alphanumeric ImageId.
    """
    while True:
        # Generate a string with pattern: 2 letters + 4 digits + 2 letters
        new_id = ''.join([
            random.choice(string.ascii_uppercase) for _ in range(2)
        ] + [
            random.choice(string.digits) for _ in range(4)
        ] + [
            random.choice(string.ascii_uppercase) for _ in range(2)
        ])
        
        # Check if the generated ID is unique
        # If it is not in the set of existing image names, return it
        # If it is in the set, generate a new one
        if new_id not in all_image_names:
            return new_id

def read_colmap_model(colmap_path):
    """Read COLMAP model using read_write_model functions.
    
    Args:
        colmap_path (str): Path to the COLMAP model directory.
    
    Returns:
        tuple: A tuple containing cameras, images, and points3D.
    """
    try:
        # Read the COLMAP model
        cameras, images, points3D = read_model(colmap_path, ext='.bin')
        print(f"Read COLMAP model with {len(cameras)} cameras, {len(images)} images, and {len(points3D)} 3D points")
        
        return cameras, images, points3D
    except Exception as e:
        print(f"ERROR: Failed to read COLMAP model: {e}")
        sys.exit(1)

def extract_image_names_from_colmap(images):
    """Extract Image names from COLMAP image names.
    
    Args:
        images (dict): Dictionary of COLMAP images.
        
    Returns:
        tuple: A tuple containing a set of unique image names and a mapping from original image names to ImageId.
    """
    image_names = set()
    # Map to store the original image name to the ImageId
    image_name_to_id_map = {}
    
    # Iterate through the images and extract the ImageId
    for colmap_image_name, image in images.items():
        name = image.name
        # Extract ImageId from the name (assuming format like "camX/XXXX_XXNNNNXX_X.jpg")
        parts = name.split('_')
        if len(parts) >= 2:
            image_name = parts[1]
            image_names.add(image_name)
            if image_name not in image_name_to_id_map:
                # Get the number in the name (e.g. 0001)
                number = int(parts[0].split('/')[-1])
                image_name_to_id_map[image_name] = number
    
    print(f"Found {len(image_names)} unique ImageIds in COLMAP data")
    return image_names, image_name_to_id_map

def is_point_in_chunk(x, y, z, center_x, center_y, center_z, extent_x, extent_y, extent_z): # NOTE: This function is not used anymore
    """Check if a point is within the chunk boundaries.
    
    Args:
        x, y, z: Coordinates of the point.
        center_x, center_y, center_z: Center coordinates of the chunk.
        extent_x, extent_y, extent_z: Extent of the chunk in each dimension.
    
    Returns:
        bool: True if the point is within the chunk, False otherwise.
    """
    # Check if the point is within the extent in all dimensions
    return (abs(x - center_x) <= extent_x/2 and 
            abs(y - center_y) <= extent_y/2 and 
            abs(z - center_z) <= extent_z/2)

def read_center_and_extent(center_path, extent_path): # NOTE: This function is not used anymore
    """Read center and extent values from files.
    
    Args:
        center_path (str): Path to the center file.
        extent_path (str): Path to the extent file.
        
    Returns:
        tuple: A tuple containing the center and extent values as lists of floats.
    """
    with open(center_path, 'r') as f:
        center = [float(val) for val in f.read().strip().split()]
    
    with open(extent_path, 'r') as f:
        extent = [float(val) for val in f.read().strip().split()]
    
    return center, extent

def read_translation_values(translation_path):
    """Read translation values from JSON file.
    
    Args:
        translation_path (str): Path to the translation JSON file.
    
    Returns:
        tuple: A tuple containing the x and y translation values as floats.
    """
    with open(translation_path, 'r') as f:
        translation = json.load(f)
    
    return translation["x_translation"], translation["y_translation"]

def compute_extrinsics(face, vehicle_direction, yaw):
    """Compute extrinsics for a given face, vehicle direction, and yaw.
    
    Args:
        face (str): The face direction ('f1', 'r1', 'b1', 'l1', etc.).
        vehicle_direction (float): The vehicle direction in degrees.
        yaw (float): The yaw in degrees.
        
    Returns:
        np.ndarray: The rotation matrix for the given face, vehicle direction, and yaw.
    """
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
        rotation_matrix (np.ndarray): The rotation matrix.
        position (list or np.ndarray): The position vector.
    
    Returns:
        np.ndarray: The translation vector.
    """
    position = np.array(position, dtype=float).reshape(3, 1)
    
    # Compute the translation vector
    translation = -np.dot(rotation_matrix, position)
    
    return translation.flatten()

def parse_iso_timestamp(timestamp_str):
    """
    Parse ISO format timestamp strings that may have variable precision in fractional seconds.
    
    Args:
        timestamp_str (str): ISO format timestamp string, e.g. "2023-10-23T10:30:32.24Z"
        
    Returns:
        datetime: Parsed datetime object
    """
    # Check if the timestamp ends with 'Z' and replace it with '+00:00'
    if timestamp_str.endswith('Z'):
        timestamp_str = timestamp_str[:-1] + "+00:00"
    
    # If there's no timezone information, add '+00:00'
    if not (timestamp_str.endswith('+00:00') or timestamp_str.endswith('-00:00')):
        timestamp_str += "+00:00"
    
    # Handle the case where we might have less than 6 digits for microseconds
    if '.' in timestamp_str:
        # Split at the decimal point
        main_part, frac_part = timestamp_str.split('.')
        
        # Split the fractional part at the timezone indicator if present
        if '+' in frac_part:
            frac_digits, timezone = frac_part.split('+', 1)
            frac_part = frac_digits.ljust(6, '0') + '+' + timezone
        elif '-' in frac_part:
            frac_digits, timezone = frac_part.split('-', 1)
            frac_part = frac_digits.ljust(6, '0') + '-' + timezone
        else:
            frac_part = frac_part.ljust(6, '0')
        
        timestamp_str = f"{main_part}.{frac_part}"
    
    return datetime.fromisoformat(timestamp_str)

def interpolate_recordings(json_data, all_image_names, image_name_to_id_map, translation, faces):
    """
    Process the recordings and insert new ones based on multiple criterias.
    Filter recordings to include only those present in COLMAP
    but check against all COLMAP image IDs for uniqueness.
    Keep only augmented images within chunk boundaries.

    Args:
        json_data (dict): JSON data containing recording properties.
        all_image_names (set): Set of all image names.
        image_name_to_id_map (dict): Mapping from image names to IDs.
        translation (tuple): Translation values for x and y axes.
        faces (list): List of camera faces to consider.
    Returns:
        tuple: A tuple containing new recordings, augmented COLMAP images, updated all_image_names, and image_name_to_id_map.
    """

    recordings = json_data["RecordingProperties"]

    # Sort recordings by timestamp
    recordings.sort(key=lambda x: parse_iso_timestamp(x['RecordingTimeGps']))
    
    new_recordings = []
    augmented_colmap_images = {}
    x_translation, y_translation = translation

    colmap_position_number = 0
    # Get highest number in the map
    cyclomedia_station_number = max(image_name_to_id_map.values(), default=0) + 1

    skipped_distance = 0
    
    for i in range(len(recordings) - 1):
        current = recordings[i]
        next_rec = recordings[i + 1]
        
        # Calculate distance between points
        distance = calculate_distance(current["X"], current["Y"], next_rec["X"], next_rec["Y"])
        
        # Check if they meet our criteria for inserting a new recording
        if distance < 10:
            # Generate a new unique ImageId
            new_image_name = generate_unique_image_name(all_image_names)
            
            # Create a new recording with averaged values
            new_recording = {}
            for key in current.keys():
                if key == "ImageId":
                    new_recording[key] = new_image_name
                elif key == "ImageFolder":
                    # Keep the same image folder
                    new_recording[key] = current[key]
                elif key == "Dataset":
                    # Keep the same dataset
                    new_recording[key] = current[key]
                elif key == "DepthmapParamsRef" or key == "ImageParamsRef" or key == "CameraParamsRef":
                    # Keep the same reference parameters
                    new_recording[key] = current[key]
                elif key == "IsInAreaBuffer":
                    # Keep logical values the same
                    new_recording[key] = current[key]
                elif key == "RecordingTimeGps":
                    # Calculate middle timestamp
                    time1 = parse_iso_timestamp(current['RecordingTimeGps'])
                    time2 = parse_iso_timestamp(next_rec['RecordingTimeGps'])
                    time_diff = (time2 - time1).total_seconds() / 2
                    middle_time = time1.timestamp() + time_diff
                    middle_datetime = datetime.fromtimestamp(middle_time).isoformat().replace('+00:00', 'Z')
                    new_recording[key] = middle_datetime
                else:
                    # For all numeric values, take the average
                    try:
                        new_recording[key] = (current[key] + next_rec[key]) / 2
                    except (TypeError, ValueError):
                        # If not numeric, keep the current value
                        new_recording[key] = current[key]
        
            # Add the new recording and image_id to the collections
            new_recordings.append(new_recording)
            all_image_names.add(new_image_name)
            
            # Prepare COLMAP data for this augmented image
            # We'll create entries for each face
            for face in faces:
                # Compute extrinsics
                rotation_matrix = compute_extrinsics(face, new_recording["VehicleDirection"], new_recording["Yaw"])
                position = [new_recording["X"] - x_translation, new_recording["Y"] - y_translation, new_recording["Height"]]
                translation_vector = calculate_translation(rotation_matrix, position)
                
                # Get camera number for the current face (for naming only)
                if args.directions == '1':
                    cam_n = {
                        'f1': 1, 'r1': 2, 'b1': 3, 'l1': 4
                    }[face]
                elif args.directions == '2' or args.directions == '3':
                    cam_n = {
                        'f1': 1, 'f2': 2, 'r1': 3, 'r2': 4, 'b1': 5, 'b2': 6, 'l1': 7, 'l2': 8, 'u1': 9, 'u2': 10
                    }[face]
                elif args.directions == '4':
                    cam_n = {
                        'f1': 1, 'r1': 2, 'b1': 3, 'l1': 4, 'u1': 5, 'u2': 6
                    }[face]
            
                # Create a unique index for this augmented image if it does not already have an assigned number
                if new_image_name not in image_name_to_id_map:
                    image_name_to_id_map[new_image_name] = cyclomedia_station_number
                    cyclomedia_station_number += 1
                    idx_str = f"{image_name_to_id_map[new_image_name]:04d}"
                else:
                    idx_str = f"{image_name_to_id_map[new_image_name]:04d}"
                image_name = f"cam{cam_n}/{idx_str}_{new_image_name}_{face}.jpg"
                
                # Convert rotation matrix to quaternion
                qvec = rotmat2qvec(rotation_matrix)
                
                # Create a COLMAP Image object that's compatible with write_model
                augmented_colmap_images[colmap_position_number] = Image(
                    id=colmap_position_number,
                    qvec=qvec,
                    tvec=translation_vector,
                    camera_id=1,  # Use camera ID 1 for all images
                    name=image_name,
                    xys=np.array([], dtype=np.float64).reshape(0, 2),
                    point3D_ids=np.array([], dtype=np.int64)
                )
                
                colmap_position_number += 1
        else:
            skipped_distance+=1
    
    print(f"Added {len(new_recordings)} new recordings. Skipped {skipped_distance} due to distance constraints")
    print(f"Created {len(augmented_colmap_images)} COLMAP image entries for augmented images")
    
    return new_recordings, augmented_colmap_images

def write_colmap_images(output_path, images_dict):
    """Write COLMAP images_depths.bin file.
    
    Args:
        output_path (str): Path to the output .bin file.
        images_dict (dict): Dictionary of COLMAP images.
    """
    try:
        remapped_images = {}
        for new_id, (_, image) in enumerate(sorted(images_dict.items())):
            # Create a copy of the image with updated ID
            updated_image = Image(
                id=new_id,
                qvec=image.qvec,
                tvec=image.tvec,
                camera_id=image.camera_id,
                name=image.name,
                xys=image.xys,
                point3D_ids=image.point3D_ids
            )
            remapped_images[new_id] = updated_image
        
        # Write the remapped images to a .bin file
        write_images_binary(remapped_images, output_path)
        print(f"Successfully wrote {len(images_dict)} augmented images to {output_path}")
        
    except Exception as e:
        print(f"ERROR: Failed to write COLMAP images: {e}")
        sys.exit(1)

def process_model():
    """Process a single chunk. For each chunk, read the COLMAP model,
    extract image names, and interpolate recordings.
    Filter recordings to include only those present in COLMAP for this chunk
    but check against all COLMAP image IDs for uniqueness.
    Keep only augmented images within chunk boundaries.
    Write COLMAP images_depths.bin for this chunk.
    Handle exceptions for robustness.
    
    Returns:
        tuple: A tuple containing new recordings, augmented COLMAP images
    """
    try:
        aligned_colmap_path = os.path.join(args.project_dir, 'camera_calibration', 'aligned', 'sparse', '0')
        aligned_images_depths_path = os.path.join(aligned_colmap_path, 'images_depths.bin')
        output_path = os.path.join(args.project_dir, 'ss_raw_images', 'recording_details_augmented.json')
        recording_details_path = os.path.join(args.project_dir, 'ss_raw_images', 'recording_details_train.json')
        translation_json_path = os.path.join(args.project_dir, 'camera_calibration', 'translation.json')

        # Define the faces based on chosen directions
        if args.directions == '1':
            faces = ['f1', 'r1', 'b1', 'l1']
        elif args.directions == '2':
            faces = ['f1', 'f2', 'r1', 'r2', 'b1', 'b2', 'l1', 'l2']
        elif args.directions == '3':
            faces = ['f1', 'f2', 'r1', 'r2', 'b1', 'b2', 'l1', 'l2', 'u1', 'u2']
        elif args.directions == '4':
            faces = ['f1', 'r1', 'b1', 'l1', 'u1', 'u2']

        # Collect all image IDs (both train and test) from the entire model first
        all_image_names, image_name_to_id_map = collect_all_image_names(args.project_dir)

        # Read the input JSON file (just data of train images)
        with open(recording_details_path, 'r') as f:
            data = json.load(f)

        # Read translation values
        translation = read_translation_values(translation_json_path)
        
        # Process the recordings for this chunk
        new_recordings, augmented_colmap_images = interpolate_recordings(
            data, all_image_names, image_name_to_id_map, translation, faces
        )
        
        # Write COLMAP images_depths.bin for this chunk
        write_colmap_images(aligned_images_depths_path, augmented_colmap_images)

         # Add all new recordings to the original data
        data["RecordingProperties"].extend(new_recordings)
        
        # Write the updated data to the output file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return
        
    except Exception as e:
        print(f"Error processing model: {e}")
        return 

def collect_all_image_names(project_dir):
    """Collect all image names from the entire model.
    
    Args:
        project_dir (str): Path to the project directory.
    
    Returns:
        tuple: A tuple containing a set of all image names and a mapping from original image names to ImageId.
    """
    aligned_path = os.path.join(project_dir, 'camera_calibration', 'aligned', 'sparse', '0')
    
    all_image_names = set()
    
    # Read from aligned model
    try:
        cameras, images, points3D = read_colmap_model(aligned_path)
        aligned_names, image_name_to_id_map = extract_image_names_from_colmap(images)
        all_image_names.update(aligned_names)
    except Exception as e:
        print(f"Error reading aligned model: {e}")
    
    return all_image_names, image_name_to_id_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment recordings and create COLMAP images_depths.bin')
    parser.add_argument('--project_dir', type=str, required=True, help='Path to project directory')
    parser.add_argument('--directions', type=str, default='3', choices=['1', '2', '3', '4'], 
                        help='Camera directions: 1=FRLB, 2=F1F2R1R2B1B2L1L2, 3=F1F2R1R2B1B2L1L2U1U2')
    
    args = parser.parse_args()
    process_model()