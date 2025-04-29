'''
Thesis Project: Street-sparse-3DGS
Author: Iacopo Ermacora
Date: 11/2024-06/2025

Description: This script is used to save the output of the Street-sparse-3DGS project for a test.
'''

import os
import shutil
import datetime
import argparse

def get_loc_and_desc():
    """
    Get user input for location name and description.

    Returns:
        tuple: A tuple containing the location name and description.
    """
    location_name = input("Enter the location name: ")
    description = input("Enter a short description of the capture: ")
    return location_name, description

def copy_folders(location_name, description, path):
    """
    Copy the specified folders to a new directory with a timestamp.

    Args:
        location_name (str): The name of the location.
        description (str): A short description of the capture.
        path (str): The path to the directory where the folders will be copied.
    """
    # Get today's date in the format YYYYMMDD
    today_date = datetime.datetime.today().strftime('%Y%m%d')
    
    # Define the new folder name
    new_folder_name = f"3DGS_{location_name}_{today_date}"

    new_folder = os.path.join(path, new_folder_name)
    
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

def list_folders(path):
    """
    List all folders in the specified storage path.

    Args:
        path (str): The path to the directory to list.
    Returns:
        list: A list of folder names in the storage path.
    """
    try:
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        if not folders:
            print(f"No folders found in {path}.")
        return folders
    except FileNotFoundError:
        print(f"The path '{path}' does not exist.")
        return []

def main():
    location_name, description = get_loc_and_desc()

    # List the available folders in the storage path
    available_datasets = list_folders(args.storage_path)

    if not available_datasets:
        print("There are no folder available, exiting script...")
        return  # No folders available, exit the script

    # Display the available folders and prompt user to select one
    print(f"Available datasets in {args.storage_path}:")
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
    if not os.path.exists(os.path.join(args.storage_path, selected_dataset)):
        print(f"The directory '{os.path.join(args.storage_path, selected_dataset)}' does not exist. Please check the path and try again.")
        return
    else:
        copy_folders(location_name, description, os.path.join(args.storage_path, selected_dataset))

# Run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True, help="Path to the project directory")
    parser.add_argument('--storage_path', type=str, default="/media/nfs/9_1_raid/iermacora/Street-sparse-3DGS_outputs", help="Path to the storage directory")
    args = parser.parse_args()
    main()
