import os
import shutil

# Function to copy the folders
def copy_folders(source_dir):
    
    folders_to_copy = ["camera_calibration", "output", "colmap_output_added_bin"]

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    if not os.path.exists(os.path.join(base_path, "ss_raw_images")) or not os.path.exists(os.path.join(base_path, "inputs")):
        print("Error: the 'ss_raw_images' and 'inputs' folders do not exist in the current directory. Please make sure to have the correct folder structure before importing data.")
        return

    # Check if the folders already exist in the current directory
    for folder in folders_to_copy:
        if os.path.exists(os.path.join(base_path, folder)):
            # Throw an error if the folder already exists
            raise Exception(f"Folder '{folder}' already exists in the current directory, please clean the directory before moving in new data...")
        # Check if the source folder exists
        if not os.path.exists(os.path.join(source_dir, folder)):
            # Throw an error if the folder already exists
            raise Exception(f"Folder '{folder}' does not exists in the storage, please make sure the data is shaped correctly before importing...")
    for folder in folders_to_copy:
        print(f"Copying folder '{folder}' from {source_dir} to the current directory.")
        # Copy the folder to the current directory
        shutil.copytree(os.path.join(source_dir, folder), os.path.join(base_path, folder))
        print(f"Copied folder '{folder}' from {source_dir} to the current directory.")

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
    # Prompt the user to input the source directory (where the folders are stored)
    storage_path = "/media/raid_1/iermacora/Street-sparse-3DGS_outputs/"

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
        # List the available folders in the storage path
        available_folders = list_folders(os.path.join(storage_path, selected_dataset))

        if not available_folders:
            print("There are no tests available for this dataset, exiting script...")
            return  # No folders available, exit the script

        # Display the available folders and prompt user to select one
        print(f"Available tests in {selected_dataset}:")
        for idx, folder in enumerate(available_folders, 1):
            print(f"{idx}. {folder}")
        
        # Get user selection
    try:
        selection = int(input(f"Select a test number (1-{len(available_folders)}): "))
        if selection < 1 or selection > len(available_folders):
            print("Invalid selection. Please choose a valid number.")
            return
        selected_folder = available_folders[selection - 1]
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return
    
    copy_folders(os.path.join(os.path.join(storage_path, selected_dataset), selected_folder))

# Run the script
if __name__ == "__main__":
    main()
