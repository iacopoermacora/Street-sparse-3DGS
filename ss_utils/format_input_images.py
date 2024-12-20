import os
from pyproj import Transformer
import shutil
from PIL import Image
import json
from datetime import datetime
import piexif
from PIL.ExifTags import TAGS, GPSTAGS
from fractions import Fraction

# Step 1: Parse JSON to get image info and timestamps
def parse_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    recordings = data['RecordingProperties']
    image_info = {}
    
    # Extract relevant info from the json (ImageId, timestamp, GPS coordinates)
    for record in recordings:
        image_info[record['ImageId']] = {
            'timestamp': record['RecordingTimeGps'],
            'latitude': record['Y'],
            'longitude': record['X'],
        }
    
    return image_info

# Step 2: Sort images based on timestamp
def sort_images_by_time(image_info):
    sorted_images = sorted(image_info.items(), key=lambda x: datetime.fromisoformat(x[1]['timestamp'].replace("Z", "+00:00")))
    return sorted_images

# Step 4: Copy and rename images
def copy_and_rename_images(images, base_path):
    if not os.path.exists(os.path.join(base_path, "inputsB/images")):
        os.makedirs(os.path.join(base_path, "inputsB/images"))
        inputs_image_folder = os.path.join(base_path, "inputsB/images")
    else:
        # Throw an error if the folder already exists
        raise Exception("Input folder already exists. Please remove or rename the existing folder.")
    
    # Define the directions
    directions = ['f', 'b', 'l', 'r']

    for idx, (image_id, details) in enumerate(images):
        idx = str(idx).zfill(4)
        lat = details['latitude']
        lon = details['longitude']
        
        for camNum, direction in enumerate(directions):
            # Get the new filename with incremented number
            new_name = f"{idx}_{image_id}_{direction}.jpg"
            
            # Get image file path (you need to modify this to the actual location of your images)
            image_path = f"{base_path}/ss_raw_images/images/level_2/color/{image_id}_{direction}.jpg"
            
            # Determine the camera folder
            cam_folder = os.path.join(inputs_image_folder, f"cam{camNum+1}")

            if not os.path.exists(cam_folder):
                os.makedirs(cam_folder)
            
            # Copy image to the new folder with the new name
            shutil.copy(image_path, os.path.join(cam_folder, new_name))
            
            # Step 5: Update EXIF data
            update_exif(os.path.join(cam_folder, new_name), lat, lon)

def convert_to_dms(decimal_degrees):
    """Convert decimal degrees to degrees, minutes, seconds format"""
    degrees = int(decimal_degrees)
    minutes = int((decimal_degrees - degrees) * 60)
    seconds = ((decimal_degrees - degrees) * 60 - minutes) * 60
    
    # Convert to EXIF format (rationals)
    degrees = (degrees, 1)
    minutes = (minutes, 1)
    seconds = (int(seconds * 10000), 10000)
    
    return degrees, minutes, seconds

def convert_coordinates(x, y):
    """Convert RD (Dutch) coordinates to WGS84"""
    transformer = Transformer.from_crs("EPSG:28992", "EPSG:4326")
    lat, lon = transformer.transform(x, y)
    return lat, lon

def update_exif(image_path, lat, lon):
    """Add GPS information to image EXIF data"""

    # Convert coordinates to WGS84
    lat, lon = convert_coordinates(lat, lon)

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

# Step 7: Main Function to execute the script
def main(json_file, base_path):
    image_info = parse_json(json_file)
    sorted_images = sort_images_by_time(image_info)
    copy_and_rename_images(sorted_images, base_path)

# Call the main function
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    json_file = os.path.join(base_path, "ss_raw_images/recording_details.json")
    main(json_file, base_path)

