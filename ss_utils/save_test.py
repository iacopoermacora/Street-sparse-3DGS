import os
import shutil
import datetime

# Function to prompt user for input
def get_user_input():
    location_name = input("Enter the location name: ")
    description = input("Enter a short description of the capture: ")
    return location_name, description

# Function to copy folders and create the new structure
def copy_folders(location_name, description, storage_path):
    # Get today's date in the format YYYYMMDD
    today_date = datetime.datetime.today().strftime('%Y%m%d')
    
    # Define the new folder name
    new_folder_name = f"3DGS_{location_name}_{today_date}"

    new_folder = os.path.join(storage_path, new_folder_name)
    
    # Define the list of folders to copy
    folders_to_copy = ["camera_calibration", "inputs", "output", "ss_raw_images", "colmap_output"]
    
    # Create the new directory
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # Get current project root
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Copy each folder to the new directory
    for folder in folders_to_copy:
        source_folder = os.path.join(base_path, folder)
        destination_folder = os.path.join(new_folder, folder)
        if os.path.exists(source_folder):
            print(f"Copying folder {folder} to {new_folder}, this might take a while...")
            shutil.copytree(source_folder, destination_folder)
            print(f"Copied folder {folder} to {new_folder}")
        else:
            print(f"Folder {folder} not found, skipping...")

    # Create info.txt with the description inside the new folder
    info_file_path = os.path.join(new_folder, "info.txt")
    with open(info_file_path, "w") as f:
        f.write(f"Description: {description}\n")
    print(f"Description written to {info_file_path}")

    # Ensure ss_raw_images exists in the original directory
    os.makedirs(os.path.join(base_path, "ss_raw_images"), exist_ok=True)
    print("Ensured ss_raw_images folder exists in the original directory")

# Main function to run the script
def main():
    location_name, description = get_user_input()
    storage_path = "/media/raid_1/iermacora/Street-sparse-3DGS_outputs"
    copy_folders(location_name, description, storage_path)

# Run the script
if __name__ == "__main__":
    main()
