import os
import shutil
import datetime
import argparse

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
    folders_to_copy = ["output"]
    
    # Create the new directory
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # Copy each folder to the new directory
    for folder in folders_to_copy:
        source_folder = os.path.join(args.project_dir, folder)
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

# Main function to run the script
def main():
    location_name, description = get_user_input()
    storage_path = "/media/raid_1/iermacora/Street-sparse-3DGS_outputs"

    # List the available folders in the storage path
    available_datasets = list_folders(storage_path)

    if not available_datasets:
        print("There are no folder available, exiting script...")
        return  # No folders available, exit the script

    # Display the available folders and prompt user to select one
    print(f"Available datasets in {storage_path}:")
    for idx, folder in enumerate(available_datasets, 1):
        print(f"{idx}. {folder}")
    
    # Get user selection
    try:
        selection = int(input(f"Select a dataset number (1-{len(available_datasets)}): "))
        if selection < 1 or selection > len(available_datasets):
            print("Invalid selection. Please choose a valid number.")
            return
        selected_dataset = available_datasets[selection - 1]
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return

    # Validate the provided path
    if not os.path.exists(os.path.join(storage_path, selected_dataset)):
        print(f"The directory '{os.path.join(storage_path, selected_dataset)}' does not exist. Please check the path and try again.")
        return
    else:
        copy_folders(location_name, description, os.path.join(storage_path, selected_dataset))

# Run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True, help="Path to the project directory")
    args = parser.parse_args()
    main()
