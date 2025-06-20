'''
Thesis Project: Street-sparse-3DGS
Author: Iacopo Ermacora
Date: 11/2024-06/2025

Description: This script processes images from a JSON file, sorts them by timestamp,
copies them to a new directory, renames them, and updates their EXIF data with GPS coordinates.
This script is designed to work with images from a cyclomedia dataset and may require adjustments
for different datasets or directory structures.
'''

import os
from pyproj import Transformer
import shutil
from PIL import Image
import json
from datetime import datetime
import piexif
from PIL.ExifTags import TAGS, GPSTAGS
from fractions import Fraction
from tqdm import tqdm
import argparse

def parse_json(json_file):
    """Parse the JSON file to extract image information
    
    Args:
        json_file (str): Path to the JSON file containing image metadata.
    
    Returns:
        dict: A dictionary containing image IDs and their corresponding metadata.
    """

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

# Step 4: Copy and rename images
def copy_and_rename_images(images, raw_image_input, directions):
    """Copy and rename images based on their metadata
    Args:
        images (list): List of tuples containing image IDs and their metadata.
        raw_image_input (str): Path to the raw image input directory.
        directions (str): Camera directions to be used for renaming.
    """

    print("Copying images and converting GPS coordinates")

    if not os.path.exists(os.path.join(args.project_dir, "inputs/images")):
        os.makedirs(os.path.join(args.project_dir, "inputs/images"))
        inputs_image_folder = os.path.join(args.project_dir, "inputs/images")
    else:
        # Throw an error if the folder already exists
        raise Exception("Input folder already exists. Please remove the existing folder.")

    for idx, (image_id, details) in tqdm(enumerate(images), total=len(images)):
        idx = str(idx).zfill(4)
        x = details['x']
        y = details['y']
        
        for camNum, direction in enumerate(directions):
            # Get the new filename with incremented number
            new_name = f"{idx}_{image_id}_{direction}.jpg"
            
            # Get image file path (you need to modify this to the actual location of your images)
            image_path = f"{raw_image_input}/{image_id}_{direction}.jpg"
            
            # Determine the camera folder
            cam_folder = os.path.join(inputs_image_folder, f"cam{camNum+1}")

            if not os.path.exists(cam_folder):
                os.makedirs(cam_folder)
            
            # Copy image to the new folder with the new name
            shutil.copy(image_path, os.path.join(cam_folder, new_name))

# Call the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True, help="Path to the project directory")
    parser.add_argument('--directions', type=str, default="3", choices=['1', '2', '3', '4'], help="Camera directions: 1=Front, Right, Back, Left; 2=Front1, Front2, Right1, Right2, Back1, Back2, Left1, Left2; 3=Front1, Front2, Right1, Right2, Back1, Back2, Left1, Left2, Up1, Up2")
    args = parser.parse_args()

    raw_input = f"{args.project_dir}/ss_raw_images"
    raw_image_input = f"{raw_input}/images/level_2/color"
    json_file_train = os.path.join(args.project_dir, "camera_calibration", "extras", "recording_details_train.json")
    json_file_test = os.path.join(args.project_dir, "camera_calibration", "extras", "recording_details_test.json")
    image_info_train = parse_json(json_file_train)
    image_info_test = parse_json(json_file_test)
    # Merge the two dictionaries
    image_info = {**image_info_train, **image_info_test}
    # Sort images by timestamp
    sorted_images = sort_images_by_time(image_info)

    if args.directions == "1":
        face_directions = ["f1", "r1", "b1", "l1"]
    elif args.directions == "2":
        face_directions = ["f1", "f2", "r1", "r2", "b1", "b2", "l1", "l2"]
    elif args.directions == "3":
        face_directions = ["f1", "f2", "r1", "r2", "b1", "b2", "l1", "l2", "u1", "u2"]
    elif args.directions == "4":
        face_directions = ["f1", "r1", "b1", "l1", "u1", "u2"]

    copy_and_rename_images(sorted_images, raw_image_input, face_directions)

