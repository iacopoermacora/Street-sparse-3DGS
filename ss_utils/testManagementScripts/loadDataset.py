import os
import shutil
import argparse

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

# Function to copy the folders
def copy_folders(source_dir):
    
    folders_to_copy = ["ss_raw_images", "inputs"]

    # Check if the folders already exist in the current directory
    for folder in folders_to_copy:
        if os.path.exists(os.path.join(args.project_dir, folder)):
            # Throw an error if the folder already exists
            raise Exception(f"Folder '{folder}' already exists in the current directory, please clean the directory before moving in new data...")
        # Check if the source folder exists
        if not os.path.exists(os.path.join(source_dir, folder)):
            # Throw an error if the folder already exists
            raise Exception(f"Folder '{folder}' does not exists in the storage, please make sure the data is shaped correctly before importing...")
    for folder in folders_to_copy:
        print(f"Copying folder '{folder}' from {source_dir} to the current directory.")
        # Copy the folder to the current directory
        shutil.copytree(os.path.join(source_dir, folder), os.path.join(args.project_dir, folder))
        print(f"Copied folder '{folder}' from {source_dir} to the current directory.")
    
# Main function to run the script
def main():
    # Prompt the user to input the source directory (where the folders are stored)
    storage_path = "/media/raid_1/iermacora/Street-sparse-3DGS_outputs/"

    # List the available folders in the storage path
    available_datasets = list_folders(storage_path)

    if not available_datasets:
        print("There are no datasets available, exiting script...")
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
        copy_folders(os.path.join(os.path.join(storage_path, selected_dataset), "Dataset"))

# Run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True, help="Path to the project directory")
    args = parser.parse_args()
    main()