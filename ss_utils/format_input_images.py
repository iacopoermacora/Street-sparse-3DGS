import os
import shutil
from PIL import Image
import json
from datetime import datetime
import piexif

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
            image_path = f"base_path/ss_raw_images/level_2/color/{image_id}_{direction}.jpg"
            
            # Determine the camera folder
            cam_folder = os.path.join(inputs_image_folder, f"cam{camNum+1}")

            if not os.path.exists(cam_folder):
                os.makedirs(cam_folder)
            
            # Copy image to the new folder with the new name
            shutil.copy(image_path, os.path.join(cam_folder, new_name))
            
            # Step 5: Update EXIF data
            update_exif(os.path.join(cam_folder, new_name), lat, lon)

# Step 6: Update EXIF data with GPS coordinates
def update_exif(image_path, latitude, longitude):
    """
    Update GPS EXIF data in an image.
    
    Args:
        image_path (str): Path to the image file.
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
    """
    def decimal_to_dms(decimal):
        """Convert decimal coordinates to degrees, minutes, and seconds."""
        degrees = int(decimal)
        minutes = int((decimal - degrees) * 60)
        seconds = round((decimal - degrees - minutes / 60) * 3600, 6)
        return degrees, minutes, seconds

    def to_exif_rational(number):
        """Convert a number to EXIF rational format (numerator/denominator)."""
        return (int(number * 1000000), 1000000)

    # Convert latitude and longitude to DMS
    lat_dms = decimal_to_dms(abs(latitude))
    lon_dms = decimal_to_dms(abs(longitude))
    
    # Determine the references (N/S, E/W)
    lat_ref = b"N" if latitude >= 0 else b"S"
    lon_ref = b"E" if longitude >= 0 else b"W"
    
    # Prepare GPS IFD dictionary
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: lat_ref,
        piexif.GPSIFD.GPSLatitude: [to_exif_rational(v) for v in lat_dms],
        piexif.GPSIFD.GPSLongitudeRef: lon_ref,
        piexif.GPSIFD.GPSLongitude: [to_exif_rational(v) for v in lon_dms],
    }
    
    # Load existing EXIF data and update GPS IFD
    exif_dict = piexif.load(image_path)
    exif_dict["GPS"] = gps_ifd
    exif_bytes = piexif.dump(exif_dict)
    
    # Save updated EXIF data back to the image
    image = Image.open(image_path)
    image.save(image_path, exif=exif_bytes)

# Step 7: Main Function to execute the script
def main(json_file, base_path):
    image_info = parse_json(json_file)
    sorted_images = sort_images_by_time(image_info)
    copy_and_rename_images(sorted_images, base_path)

# Call the main function
if __name__ == "__main__":
    base_path = os.path.join(os.path.abspath(__file__), "..")
    json_file = os.path.join(base_path, "ss_raw_images/recording_details.json")
    main(json_file, base_path)

