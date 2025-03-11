import os
import shutil
import datetime

# Function to prompt user for input
def get_user_input():
    location_name = input("Enter the location name: ")
    description = input("Enter a short description of the dataset: ")
    return location_name, description

# Function to list available folders in the storage path
def list_folders(storage_path):
    # Get a list of directories in the storage path
    try:
        folders = [f for f in os.listdir(storage_path) if os.path.isdir(os.path.join(storage_path, f))]
        if not folders:
            print(f"No folders found in {storage_path}.")
        return folders
    except FileNotFoundError:
        print(f"The path '{storage_path}' does not exist.")
        return []
    
# Function to copy folders and create the new structure
def copy_folders(location_name, description, storage_path):
    # Get today's date in the format YYYYMMDD
    today_date = datetime.datetime.today().strftime('%Y%m%d')

    new_folder = os.path.join(storage_path, location_name)
    dataset_folder = os.path.join(new_folder, "Dataset")
    
    # Define the list of folders to copy
    folders_to_copy = ["ss_raw_images", "inputs"]
    
    # Create the new directory
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        os.makedirs(dataset_folder)

    # Get current project root
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    # Copy each folder to the new directory
    for folder in folders_to_copy:
        source_folder = os.path.join(base_path, folder)
        destination_folder = os.path.join(dataset_folder, folder)
        if os.path.exists(source_folder):
            print(f"Copying folder {folder} to {dataset_folder}, this might take a while...")
            shutil.copytree(source_folder, destination_folder)
            print(f"Copied folder {folder} to {dataset_folder}")
        else:
            print(f"Folder {folder} not found, skipping...")

    # Create info.txt with the description inside the new folder
    info_file_path = os.path.join(dataset_folder, "info.txt")
    with open(info_file_path, "w") as f:
        f.write(f"Dataset description: {description}\n")
    print(f"Description written to {info_file_path}")
    
# Main function to run the script
def main():
    location_name, description = get_user_input()
    storage_path = "/media/raid_1/iermacora/Street-sparse-3DGS_outputs"

    if not os.path.exists(os.path.join(storage_path, location_name)):
        copy_folders(location_name, description, storage_path)
    else:
        print(f"Error: folder '{location_name}' already exists in the storage path, please choose a different name.")

# Run the script
if __name__ == "__main__":
    main()