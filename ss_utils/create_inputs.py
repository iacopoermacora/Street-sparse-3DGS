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

def sort_images_by_time(image_info):
    """Sort images by timestamp
    Args:
        image_info (dict): Dictionary containing image IDs and their metadata.
    
    Returns:
        list: A sorted list of tuples containing image IDs and their metadata.
    """
    print("Sorting images...")

    sorted_images = sorted(image_info.items(), key=lambda x: datetime.fromisoformat(x[1]['timestamp'].replace("Z", "+00:00")))
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
            
            # Step 5: Update EXIF data
            update_exif(os.path.join(cam_folder, new_name), x, y)

def convert_to_dms(decimal_degrees):
    """Convert decimal degrees to degrees, minutes, seconds format
    
    Args:
        decimal_degrees (float): Decimal degrees to convert.
        
    Returns:
        tuple: A tuple containing degrees, minutes, and seconds as fractions.
    """
    degrees = int(decimal_degrees)
    minutes = int((decimal_degrees - degrees) * 60)
    seconds = ((decimal_degrees - degrees) * 60 - minutes) * 60
    
    # Convert to EXIF format (rationals)
    degrees = (degrees, 1)
    minutes = (minutes, 1)
    seconds = (int(seconds * 10000), 10000)
    
    return degrees, minutes, seconds

def convert_coordinates(x, y):
    """Convert RD (Dutch) coordinates to WGS84
    Args:
        x (float): RD x-coordinate.
        y (float): RD y-coordinate.
    
    Returns:
        tuple: Latitude and longitude in WGS84 format.
    """
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326")
    lat, lon = transformer.transform(y, x)
    return lat, lon

def update_exif(image_path, x, y):
    """Add GPS information to image EXIF data
    
    Args:
        image_path (str): Path to the image file.
        x (float): Longitude coordinate.
        y (float): Latitude coordinate.
    """

    # Convert coordinates to WGS84
    lat, lon = convert_coordinates(x, y)

    # Convert decimal coordinates to DMS
    lat_dms = convert_to_dms(abs(lat))
    lon_dms = convert_to_dms(abs(lon))
    
    # Create GPS dictionary
    gps_ifd = {
        piexif.GPSIFD.GPSVersionID: (2, 2, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: 'N' if lat >= 0 else 'S',
        piexif.GPSIFD.GPSLatitude: lat_dms,
        piexif.GPSIFD.GPSLongitudeRef: 'E' if lon >= 0 else 'W',
        piexif.GPSIFD.GPSLongitude: lon_dms
    }
    
    # Create EXIF dictionary
    exif_dict = {"GPS": gps_ifd}
    
    # Convert to bytes
    exif_bytes = piexif.dump(exif_dict)
    
    # Add EXIF to image
    im = Image.open(image_path)
    im.save(image_path, exif=exif_bytes) 

# Call the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True, help="Path to the project directory")
    parser.add_argument('--directions', type=str, default="3", help="Camera directions: 1=Front, Right, Back, Left; 2=Front1, Front2, Right1, Right2, Back1, Back2, Left1, Left2; 3=Front1, Front2, Right1, Right2, Back1, Back2, Left1, Left2, Up1, Up2")
    args = parser.parse_args()

    raw_input = f"{args.project_dir}/ss_raw_images"
    raw_image_input = f"{raw_input}/images/level_2/color"
    json_file = os.path.join(raw_input, "recording_details_train_test.json")
    image_info = parse_json(json_file)
    sorted_images = sort_images_by_time(image_info)

    if args.directions == "1":
        face_directions = ["f", "r", "b", "l"]
    elif args.directions == "2":
        face_directions = ["f1", "f2", "r1", "r2", "b1", "b2", "l1", "l2"]
    elif args.directions == "3":
        face_directions = ["f1", "f2", "r1", "r2", "b1", "b2", "l1", "l2", "u1", "u2"]


    copy_and_rename_images(sorted_images, raw_image_input, face_directions)

