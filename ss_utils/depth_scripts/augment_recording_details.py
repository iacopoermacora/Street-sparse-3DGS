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

def is_point_in_chunk(x, y, z, center_x, center_y, center_z, extent_x, extent_y, extent_z):
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

def read_center_and_extent(center_path, extent_path):
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
        face (str): The face direction ('f', 'r', 'b', 'l', etc.).
        vehicle_direction (float): The vehicle direction in degrees.
        yaw (float): The yaw in degrees.
        
    Returns:
        np.ndarray: The rotation matrix for the given face, vehicle direction, and yaw.
    """
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
        rotation_matrix (np.ndarray): The rotation matrix.
        position (list or np.ndarray): The position vector.
    
    Returns:
        np.ndarray: The translation vector.
    """
    position = np.array(position, dtype=float).reshape(3, 1)
    
    # Compute the translation vector
    translation = -np.dot(rotation_matrix, position)
    
    return translation.flatten()

def interpolate_recordings(json_data, chunk_image_names, all_image_names, image_name_to_id_map, len_all_augmented_images, center, extent, translation, faces):
    """
    Process the recordings and insert new ones based on multiple criterias.
    Filter recordings to include only those present in COLMAP for this chunk
    but check against all COLMAP image IDs for uniqueness.
    Keep only augmented images within chunk boundaries.

    Args:
        json_data (dict): JSON data containing recording properties.
        chunk_image_names (set): Set of image names in the current chunk.
        all_image_names (set): Set of all image names.
        image_name_to_id_map (dict): Mapping from image names to IDs.
        len_all_augmented_images (int): Length of the augmented images dictionary.
        center (list): Center coordinates of the chunk.
        extent (list): Extent of the chunk in each dimension.
        translation (tuple): Translation values for x and y axes.
        faces (list): List of camera faces to consider.
    Returns:
        tuple: A tuple containing new recordings, augmented COLMAP images, updated all_image_names, and image_name_to_id_map.
    """
    recordings = json_data["RecordingProperties"]
    # Filter recordings to include only those present in COLMAP for this chunk
    filtered_recordings = [rec for rec in recordings if rec["ImageId"] in chunk_image_names]
    
    print(f"Filtered {len(filtered_recordings)} recordings out of {len(recordings)} total for this chunk")
    
    # Sort recordings by timestamp
    filtered_recordings.sort(key=lambda x: datetime.fromisoformat(x["RecordingTimeGps"].replace('Z', '+00:00')))
    
    # # Use all existing image IDs for uniqueness check TODO: Check if this is necessary otherwise remove
    # existing_ids = {rec["ImageId"] for rec in recordings}
    # existing_ids.update(all_image_names)
    
    new_recordings = []
    augmented_colmap_images = {}
    
    center_x, center_y, center_z = center
    extent_x, extent_y, extent_z = extent
    x_translation, y_translation = translation

    colmap_position_number = 0
    # Get highest number in the map
    colmap_station_number = max(image_name_to_id_map.values(), default=0) + 1

    skipped_out_of_bounds = 0
    skipped_distance = 0
    
    for i in range(len(filtered_recordings) - 1):
        current = filtered_recordings[i]
        next_rec = filtered_recordings[i + 1]
        
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
                    time1 = datetime.fromisoformat(current["RecordingTimeGps"].replace('Z', '+00:00'))
                    time2 = datetime.fromisoformat(next_rec["RecordingTimeGps"].replace('Z', '+00:00'))
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
            
            # Check if the new point is within the chunk boundaries
            point_x = new_recording["X"] - x_translation
            point_y = new_recording["Y"] - y_translation
            point_z = new_recording["Height"]  # Use Height as Z
            
            if is_point_in_chunk(point_x, point_y, point_z, center_x, center_y, center_z, extent_x, extent_y, extent_z):
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
                            'f': 1, 'r': 2, 'b': 3, 'l': 4
                        }[face]
                    elif args.directions == '2' or args.directions == '3':
                        cam_n = {
                            'f1': 1, 'f2': 2, 'r1': 3, 'r2': 4, 'b1': 5, 'b2': 6, 'l1': 7, 'l2': 8, 'u1': 9, 'u2': 10
                        }[face]
                    elif args.directions == '4':
                        cam_n = {
                            'f': 1, 'r': 2, 'b': 3, 'l': 4, 'u1': 5, 'u2': 6
                        }[face]
                
                    # Create a unique index for this augmented image if it does not already have an assigned number
                    if new_image_name not in image_name_to_id_map:
                        image_name_to_id_map[new_image_name] = colmap_station_number
                        colmap_station_number += 1
                        idx_str = f"{image_name_to_id_map[new_image_name]:04d}"
                    else:
                        idx_str = f"{image_name_to_id_map[new_image_name]:04d}"
                    image_name = f"cam{cam_n}/{idx_str}_{new_image_name}_{face}.jpg"
                    
                    # Convert rotation matrix to quaternion
                    qvec = rotmat2qvec(rotation_matrix)
                    
                    # Create a COLMAP Image object that's compatible with write_model
                    augmented_colmap_images[len_all_augmented_images + colmap_position_number] = Image(
                        id=len_all_augmented_images + colmap_position_number,
                        qvec=qvec,
                        tvec=translation_vector,
                        camera_id=1,  # Use camera ID 1 for all images
                        name=image_name,
                        xys=np.array([], dtype=np.float64).reshape(0, 2),
                        point3D_ids=np.array([], dtype=np.int64)
                    )
                    
                    colmap_position_number += 1
            else:
                skipped_out_of_bounds+=1
        else:
            skipped_distance+=1
    
    print(f"Added {len(new_recordings)} new recordings within chunk boundaries. Skipped {skipped_distance} due to distance and {skipped_out_of_bounds} due to out of bounds.")
    print(f"Created {len(augmented_colmap_images)} COLMAP image entries for augmented images")
    
    return new_recordings, augmented_colmap_images, all_image_names, image_name_to_id_map

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

def process_chunk(chunk_folder, translation, faces, json_data, all_image_names, image_name_to_id_map, all_augmented_images, all_new_recordings):
    """Process a single chunk. For each chunk, read the COLMAP model,
    extract image names, and interpolate recordings.
    Filter recordings to include only those present in COLMAP for this chunk
    but check against all COLMAP image IDs for uniqueness.
    Keep only augmented images within chunk boundaries.
    Write COLMAP images_depths.bin for this chunk.
    Handle exceptions for robustness.
    
    Args:
        chunk_folder (str): Path to the chunk folder.
        translation (tuple): Translation values for x and y axes.
        faces (list): List of camera faces to consider.
        json_data (dict): JSON data containing recording properties.
        all_image_names (set): Set of all image names.
        image_name_to_id_map (dict): Mapping from image names to IDs.
        all_augmented_images (dict): Dictionary of augmented COLMAP images.
        all_new_recordings (list): List of new recordings.
    
    Returns:
        tuple: A tuple containing new recordings, augmented COLMAP images, updated all_image_names, and image_name_to_id_map.
    """
    try:
        # Paths for this chunk
        colmap_bin_path = os.path.join(chunk_folder, 'sparse', '0')
        center_path = os.path.join(chunk_folder, 'center.txt')
        extent_path = os.path.join(chunk_folder, 'extent.txt')
        images_depths_bin_path = os.path.join(colmap_bin_path, 'images_depths.bin')
        
        # Read COLMAP model for this chunk
        _, images, _ = read_colmap_model(colmap_bin_path)
        
        # Extract ImageIds from COLMAP images for this chunk
        chunk_image_names, _ = extract_image_names_from_colmap(images)
        
        # Read center and extent
        center, extent = read_center_and_extent(center_path, extent_path)
        
        # Process the recordings for this chunk
        chunk_new_recordings, chunk_augmented_images, all_image_names, image_name_to_id_map = interpolate_recordings(
            json_data, chunk_image_names, all_image_names, image_name_to_id_map, len(all_augmented_images),
            center, extent, translation, faces
        )
        
        # Add to the global collections
        all_new_recordings.extend(chunk_new_recordings)
        print(f"Lenght of all_augmented_images: {len(all_augmented_images)}")
        print(f"Lenght of chunk_augmented_images: {len(chunk_augmented_images)}")
        all_augmented_images.update(chunk_augmented_images)
        print(f"After-merge of all_augmented_images: {len(all_augmented_images)}")
        
        # Write COLMAP images_depths.bin for this chunk
        write_colmap_images(images_depths_bin_path, chunk_augmented_images)
        
        print(f"Successfully processed chunk: {os.path.basename(chunk_folder)}")
        
        return all_new_recordings, all_augmented_images, all_image_names, image_name_to_id_map
        
    except Exception as e:
        print(f"Error processing chunk {os.path.basename(chunk_folder)}: {e}")
        return image_name_to_id_map

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

def main():    
    # Define the faces based on chosen directions
    if args.directions == '1':
        faces = ['f', 'r', 'b', 'l']
    elif args.directions == '2':
        faces = ['f1', 'f2', 'r1', 'r2', 'b1', 'b2', 'l1', 'l2']
    elif args.directions == '3':
        faces = ['f1', 'f2', 'r1', 'r2', 'b1', 'b2', 'l1', 'l2', 'u1', 'u2']
    elif args.directions == '4':
        faces = ['f', 'r', 'b', 'l', 'u1', 'u2']
    
    # Define paths based on project directory
    chunks_folder = os.path.join(args.project_dir, 'camera_calibration', 'chunks')
    recording_details_path = os.path.join(args.project_dir, 'ss_raw_images', 'recording_details_train.json')
    output_path = os.path.join(args.project_dir, 'ss_raw_images', 'recording_details_augmented.json')
    translation_json_path = os.path.join(args.project_dir, 'camera_calibration', 'translation.json')
    aligned_colmap_path = os.path.join(args.project_dir, 'camera_calibration', 'aligned', 'sparse', '0')
    aligned_images_depths_path = os.path.join(aligned_colmap_path, 'images_depths.bin')
    
    try:
        # Read the input JSON file (just data of train images)
        with open(recording_details_path, 'r') as f:
            data = json.load(f)
        
        # Read translation values
        translation = read_translation_values(translation_json_path)
        
        # Collect all image IDs (both train and test) from the entire model first
        all_image_names, image_name_to_id_map = collect_all_image_names(args.project_dir)
        
        # Collections for all augmented data
        all_new_recordings = []
        all_augmented_images = {}
        
        # Get all chunk folders
        chunk_folders = [os.path.join(chunks_folder, d) for d in os.listdir(chunks_folder) 
                         if os.path.isdir(os.path.join(chunks_folder, d)) and '_' in d]
        
        # Process each chunk
        for chunk_folder in tqdm(chunk_folders, desc="Processing chunks"):
            all_new_recordings, all_augmented_images, all_image_names, image_name_to_id_map = process_chunk(
                chunk_folder, translation, faces, 
                data, all_image_names, image_name_to_id_map,
                all_augmented_images, all_new_recordings
            )
        
        # Add all new recordings to the original data
        data["RecordingProperties"].extend(all_new_recordings)
        
        # Write the updated data to the output file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Write COLMAP images_depths.bin for the whole model
        write_colmap_images(aligned_images_depths_path, all_augmented_images)
        
        print(f"Successfully processed all chunks and saved to {output_path}")
        print(f"Successfully wrote COLMAP images_depths.bin to {aligned_images_depths_path}")
        print(f"Total new recordings added: {len(all_new_recordings)}")
        print(f"Total augmented COLMAP images created: {len(all_augmented_images)}")
    
    except Exception as e:
        import traceback
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment recordings and create COLMAP images_depths.bin')
    parser.add_argument('--project_dir', type=str, required=True, help='Path to project directory')
    parser.add_argument('--directions', type=str, default='3', choices=['1', '2', '3', '4'], 
                        help='Camera directions: 1=FRLB, 2=F1F2R1R2B1B2L1L2, 3=F1F2R1R2B1B2L1L2U1U2')
    
    args = parser.parse_args()
    main()