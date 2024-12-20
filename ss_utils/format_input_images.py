import os
import shutil
from PIL import Image
from PIL.ExifTags import TAGS
import json
from datetime import datetime
from geopy import distance

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

# Step 3: Create folder structure
def create_folders(base_path):
    cams = ['cam1', 'cam2', 'cam3', 'cam4']
    for cam in cams:
        cam_path = os.path.join(base_path, cam)
        if not os.path.exists(cam_path):
            os.makedirs(cam_path)

# Step 4: Copy and rename images
def copy_and_rename_images(images, image_info, base_path):
    # Create folder structure
    create_folders(base_path)
    
    # Initialize counters for each direction
    counters = {'f': 1, 'b': 1, 'r': 1, 'l': 1}
    
    for idx, (image_id, details) in enumerate(images):
        timestamp = details['timestamp']
        lat = details['latitude']
        lon = details['longitude']
        
        # Determine direction (you can modify this part if you have other criteria to determine direction)
        direction = 'f'  # Example, replace with actual logic based on image metadata
        
        # Get the new filename with incremented number
        new_name = f"{counters[direction]}_{28992}_{direction}.jpg"
        counters[direction] += 1
        
        # Get image file path (you need to modify this to the actual location of your images)
        image_path = f"images/{image_id}.jpg"
        
        # Determine the camera folder (this logic assumes the direction is the same for each image in the folder)
        cam_folder = os.path.join(base_path, f"cam{counters[direction] % 4}")
        
        # Ensure the destination folder exists
        if not os.path.exists(cam_folder):
            os.makedirs(cam_folder)
        
        # Copy image to the new folder with the new name
        shutil.copy(image_path, os.path.join(cam_folder, new_name))
        
        # Step 5: Update EXIF data
        update_exif(os.path.join(cam_folder, new_name), lat, lon)

# Step 6: Update EXIF data with GPS coordinates
def update_exif(image_path, lat, lon):
    img = Image.open(image_path)
    
    # Getting EXIF data
    exif_data = img._getexif() if img._getexif() is not None else {}
    
    # Define GPS information in EXIF format
    gps_info = {
        'GPSLatitude': lat,
        'GPSLongitude': lon
    }

    # Add GPS information to EXIF
    exif_data[TAGS['GPSLatitude']] = gps_info['GPSLatitude']
    exif_data[TAGS['GPSLongitude']] = gps_info['GPSLongitude']
    
    # Save the image with updated EXIF
    img.save(image_path, exif=exif_data)

# Step 7: Main Function to execute the script
def main(json_file, base_path="inputs"):
    image_info = parse_json(json_file)
    sorted_images = sort_images_by_time(image_info)
    copy_and_rename_images(sorted_images, image_info, base_path)

# Call the main function
if __name__ == "__main__":
    json_file = "path_to_your_json_file.json"  # Change to your actual JSON file path
    main(json_file)

