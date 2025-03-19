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
from read_write_model import Image, write_images_binary, read_model

def calculate_distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def generate_unique_image_id(existing_ids):
    """Generate a random 8-character alphanumeric ImageId that doesn't exist in existing_ids."""
    while True:
        # Generate a string with pattern: 2 letters + 4 digits + 2 letters
        new_id = ''.join([
            random.choice(string.ascii_uppercase) for _ in range(2)
        ] + [
            random.choice(string.digits) for _ in range(4)
        ] + [
            random.choice(string.ascii_uppercase) for _ in range(2)
        ])
        
        if new_id not in existing_ids:
            return new_id

def read_colmap_model(colmap_path):
    """Read COLMAP model using read_write_model functions."""
    try:
        
        # Read the COLMAP model
        cameras, images, points3D = read_model(colmap_path, ext='.bin')
        print(f"Read COLMAP model with {len(cameras)} cameras, {len(images)} images, and {len(points3D)} 3D points")
        
        return cameras, images, points3D
    except ImportError:
        print("ERROR: Could not import COLMAP's read_write_model module.")
        print("Please make sure COLMAP's python binding directory is in your PYTHONPATH.")
        print("You can add it with: import sys; sys.path.append('/path/to/colmap/scripts/python')")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read COLMAP model: {e}")
        sys.exit(1)

def extract_image_ids_from_colmap(images):
    """Extract ImageIds from COLMAP image names."""
    image_ids = set()
    
    for image_id, image in images.items():
        name = image.name
        # Extract ImageId from the name (assuming format like "camX/XXXX_XXNNNNXX_X.jpg")
        parts = name.split('_')
        if len(parts) >= 2:
            image_id = parts[1]
            image_ids.add(image_id)
    
    print(f"Found {len(image_ids)} unique ImageIds in COLMAP data")
    return image_ids

def is_point_in_chunk(x, y, z, center_x, center_y, center_z, extent_x, extent_y, extent_z):
    """Check if a point is within the chunk boundaries."""
    # Check if the point is within the extent in all dimensions
    return (abs(x - center_x) <= extent_x/2 and 
            abs(y - center_y) <= extent_y/2 and 
            abs(z - center_z) <= extent_z/2)

def read_center_and_extent(center_path, extent_path):
    """Read center and extent values from files."""
    with open(center_path, 'r') as f:
        center = [float(val) for val in f.read().strip().split()]
    
    with open(extent_path, 'r') as f:
        extent = [float(val) for val in f.read().strip().split()]
    
    return center, extent

def read_translation_values(translation_path):
    """Read translation values from JSON file."""
    with open(translation_path, 'r') as f:
        translation = json.load(f)
    
    return translation["x_translation"], translation["y_translation"]

def compute_extrinsics(face, vehicle_direction, yaw):
    """Compute extrinsics for a given face, vehicle direction, and yaw."""
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
    """Convert rotation matrix to quaternion vector."""
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

def calculate_translation(rotation_matrix, position):
    """
    Calculate the translation vector given a rotation matrix and position.
    """
    position = np.array(position, dtype=float).reshape(3, 1)
    
    # Compute the translation vector
    translation = -np.dot(rotation_matrix, position)
    
    return translation.flatten()

def interpolate_recordings(json_data, colmap_image_ids, center, extent, translation, faces):
    """
    Process the recordings and insert new ones based on the criteria.
    Filter recordings to include only those present in COLMAP and
    keep only augmented images within chunk boundaries.
    """
    recordings = json_data["RecordingProperties"]
    # Filter recordings to include only those present in COLMAP
    filtered_recordings = [rec for rec in recordings if rec["ImageId"] in colmap_image_ids]
    
    print(f"Filtered {len(filtered_recordings)} recordings out of {len(recordings)} total")
    
    # Sort recordings by timestamp
    filtered_recordings.sort(key=lambda x: datetime.fromisoformat(x["RecordingTimeGps"].replace('Z', '+00:00')))
    
    existing_ids = {rec["ImageId"] for rec in recordings}
    new_recordings = []
    augmented_colmap_images = {}
    
    center_x, center_y, center_z = center
    extent_x, extent_y, extent_z = extent
    x_translation, y_translation = translation
    
    colmap_image_id_counter = 1000000  # Start with a high number to avoid conflicts
    
    for i in range(len(filtered_recordings) - 1):
        current = filtered_recordings[i]
        next_rec = filtered_recordings[i + 1]
        
        # Calculate distance between points
        distance = calculate_distance(current["X"], current["Y"], next_rec["X"], next_rec["Y"])
        
        # Check if they meet our criteria for inserting a new recording
        if distance < 10:
            # Generate a new unique ImageId
            new_image_id = generate_unique_image_id(existing_ids)
            existing_ids.add(new_image_id)
            
            # Create a new recording with averaged values
            new_recording = {}
            for key in current.keys():
                if key == "ImageId":
                    new_recording[key] = new_image_id
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
                # Add the new recording
                new_recordings.append(new_recording)
                
                # Prepare COLMAP data for this augmented image
                # We'll create entries for each face
                for face in faces:
                    # Compute extrinsics
                    rotation_matrix = compute_extrinsics(face, new_recording["VehicleDirection"], new_recording["Yaw"])
                    position = [new_recording["X"] - x_translation, new_recording["Y"] - y_translation, new_recording["Height"]]
                    translation_vector = calculate_translation(rotation_matrix, position)
                    
                    # Get camera number for the current face
                    cam_n = {
                        'f': 1, 'r': 2, 'b': 3, 'l': 4,
                        'f1': 1, 'f2': 2, 'r1': 3, 'r2': 4, 'b1': 5, 'b2': 6, 'l1': 7, 'l2': 8, 'u1': 9, 'u2': 10
                    }[face]
                    
                    # Create a unique index for this augmented image
                    idx_str = f"aug_{len(augmented_colmap_images):04d}"
                    image_name = f"cam{cam_n}/{idx_str}_{new_image_id}_{face}.jpg"
                    
                    # Convert rotation matrix to quaternion
                    qvec = rotmat2qvec(rotation_matrix)
                    
                    # Create a COLMAP Image object that's compatible with write_model
                    # We'll use a dictionary format that can be converted to a proper Image object later
                    augmented_colmap_images[colmap_image_id_counter] = {
                        'id': colmap_image_id_counter,
                        'qvec': qvec, 
                        'tvec': translation_vector,
                        'camera_id': 1,  # Assume camera 1 is used
                        'name': image_name,
                        'xys': [],       # Empty point correspondences
                        'point3D_ids': []  # Empty 3D point IDs
                    }
                    
                    colmap_image_id_counter += 1
    
    print(f"Added {len(new_recordings)} new recordings within chunk boundaries")
    print(f"Created {len(augmented_colmap_images)} COLMAP image entries for augmented images")
    
    return new_recordings, augmented_colmap_images

def write_colmap_images_augmented(output_path, images_dict):
    """Write COLMAP images_augmented.bin file using read_write_model."""
    try:
        
        # Convert the dictionary format to proper Image objects
        images = {}
        for image_id, img_dict in images_dict.items():
            image = Image(
                id=img_dict['id'],
                qvec=img_dict['qvec'],
                tvec=img_dict['tvec'],
                camera_id=img_dict['camera_id'],
                name=img_dict['name'],
                xys=np.array(img_dict['xys'], dtype=np.float64),
                point3D_ids=np.array(img_dict['point3D_ids'], dtype=np.int64)
            )
            images[image_id] = image
        
        # Write the images to a .bin file
        write_images_binary(images, output_path)
        print(f"Successfully wrote {len(images)} augmented images to {output_path}")
        
    except ImportError:
        print("ERROR: Could not import COLMAP's read_write_model module.")
        print("Please make sure COLMAP's python binding directory is in your PYTHONPATH.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to write COLMAP images: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Augment recordings and create COLMAP images_augmented.bin')
    parser.add_argument('--recording_details', type=str, required=True, help='Path to recording_details.json')
    parser.add_argument('--output', type=str, required=True, help='Path to output recording_details_augmented.json')
    parser.add_argument('--chunk_folder', type=str, required=True, help='Path to chunk folder')
    parser.add_argument('--translation_json', type=str, required=True, help='Path to translation.json')
    parser.add_argument('--directions', type=str, default='1', choices=['1', '2', '3'], 
                        help='Camera directions: 1=FRLB, 2=F1F2R1R2B1B2L1L2, 3=F1F2R1R2B1B2L1L2U1U2')
    
    args = parser.parse_args()
    
    # Define the faces based on chosen directions
    if args.directions == '1':
        faces = ['f', 'r', 'b', 'l']
    elif args.directions == '2':
        faces = ['f1', 'f2', 'r1', 'r2', 'b1', 'b2', 'l1', 'l2']
    else:  # '3'
        faces = ['f1', 'f2', 'r1', 'r2', 'b1', 'b2', 'l1', 'l2', 'u1', 'u2']
    
    # Paths
    colmap_bin_path = os.path.join(args.chunk_folder, 'sparse', '0')
    center_path = os.path.join(args.chunk_folder, 'center.txt')
    extent_path = os.path.join(args.chunk_folder, 'extent.txt')
    images_augmented_bin_path = os.path.join(colmap_bin_path, 'images_augmented.bin')
    
    try:
        # Read the input JSON file
        with open(args.recording_details, 'r') as f:
            data = json.load(f)
        
        # Read COLMAP model
        cameras, images, points3D = read_colmap_model(colmap_bin_path)
        
        # Extract ImageIds from COLMAP images
        colmap_image_ids = extract_image_ids_from_colmap(images)
        
        # Read center and extent
        center, extent = read_center_and_extent(center_path, extent_path)
        
        # Read translation values
        translation = read_translation_values(args.translation_json)
        
        # Process the recordings
        new_recordings, augmented_colmap_images = interpolate_recordings(
            data, colmap_image_ids, center, extent, translation, faces
        )
        
        # Add new recordings to the original data
        data["RecordingProperties"].extend(new_recordings)
        
        # Write the updated data to the output file
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Write COLMAP images_augmented.bin
        write_colmap_images_augmented(images_augmented_bin_path, augmented_colmap_images)
        
        print(f"Successfully processed the recordings and saved to {args.output}")
        print(f"Successfully wrote COLMAP images_augmented.bin to {images_augmented_bin_path}")
    
    except Exception as e:
        import traceback
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()