import json
import math
import random
import string
from datetime import datetime
import os

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

def interpolate_recordings(json_data):
    """Process the recordings and insert new ones based on the criteria."""
    recordings = json_data["RecordingProperties"]
    # Sort recordings by timestamp
    recordings.sort(key=lambda x: datetime.fromisoformat(x["RecordingTimeGps"].replace('Z', '+00:00')))
    
    existing_ids = {rec["ImageId"] for rec in recordings}
    new_recordings = []
    
    for i in range(len(recordings) - 1):
        current = recordings[i]
        next_rec = recordings[i + 1]
        
        # Add the current recording to our new list
        new_recordings.append(current)
        
        # Check if these are consecutive recordings (no other recordings in between)
        time1 = datetime.fromisoformat(current["RecordingTimeGps"].replace('Z', '+00:00'))
        time2 = datetime.fromisoformat(next_rec["RecordingTimeGps"].replace('Z', '+00:00'))
        
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
                    time_diff = (time2 - time1).total_seconds() / 2
                    middle_time = time1.timestamp() + time_diff
                    middle_datetime = datetime.fromtimestamp(middle_time).isoformat().replace('+00:00', 'Z')
                    new_recording[key] = middle_datetime
                else:
                    # For all numeric values, take the average
                    new_recording[key] = (current[key] + next_rec[key]) / 2
            
            # Add the new recording
            new_recordings.append(new_recording)
    
    # Add the last recording
    if recordings:
        new_recordings.append(recordings[-1])
    
    # Update the json data with the new recordings
    json_data["RecordingProperties"] = new_recordings
    return json_data

def main():
    # Path to input and output files
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = f"{base_path}/ss_raw_images/recording_details.json"
    output_file = f"{base_path}/ss_raw_images/recording_details_augmented.json"
    
    try:
        # Read the input JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Process the recordings
        updated_data = interpolate_recordings(data)
        
        # Write the updated data to the output file
        with open(output_file, 'w') as f:
            json.dump(updated_data, f, indent=2)
        
        print(f"Successfully processed the recordings and saved to {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()