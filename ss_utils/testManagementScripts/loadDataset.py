'''
Thesis Project: Street-sparse-3DGS
Author: Iacopo Ermacora
Date: 11/2024-06/2025

Description: This script is used to load a dataset from a specified storage 
path into the current project directory.
'''

import os
import shutil
import argparse


def list_folders(path):
    """
    List all directories in the specified path.
    
    Args:
        path (str): The path to the directory to list.
        
    Returns:
        list: A list of directories in the specified path.
    """
    try:
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        if not folders:
            print(f"No folders found in {path}.")
        return folders
    except FileNotFoundError:
        print(f"The path '{path}' does not exist.")
        return []

def copy_folders(source_dir):
    """
    Copy the specified folders from the source directory to the current directory.
    
    Args:
        source_dir (str): The source directory containing the folders to copy.
    """
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
    

def main():
    # List the available folders in the storage path
    available_datasets = list_folders(args.storage_path)

    if not available_datasets:
        print("There are no datasets available, exiting script...")
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
        copy_folders(os.path.join(os.path.join(args.storage_path, selected_dataset), "Dataset"))

# Run the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True, help="Path to the project directory")
    parser.add_argument('--storage_path', type=str, default="/media/nfs/9_1_raid/iermacora/Street-sparse-3DGS_outputs/", help="Path to the storage directory")
    args = parser.parse_args()
    main()