import os
import shutil

# Function to delete folders
def delete_folders(removal):
    if removal == 'n':
        folders_to_delete = ["camera_calibration", "output", "inputs", "ss_raw_images", "colmap_output"]
    else:
        folders_to_delete = ["output", "camera_calibration", "colmap_output"]

    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    # Ask user for confirmation
    for folder in folders_to_delete:
        if os.path.exists(os.path.join(base_path, folder)):
            print(f"Removing folder '{folder}")
            # Delete the folder and its contents
            shutil.rmtree(folder)
            print(f"Folder '{folder}' has been removed.")
        else:
            print(f"Error: folder '{folder}' does now exist.")
    
    # Make the ss_raw_images folder if it does not exist
    os.makedirs(os.path.join(base_path, "ss_raw_images"), exist_ok=True)

# Main function to run the script
def main():
    # Prompt the user for confirmation to delete the folders
    confirmation = input("Do you want to remove the current test data? Make sure to have saved it before proceeding. (y/n): ").strip().lower()
    
    if confirmation == 'y':
        # Ask if they want a complete removal or just the outputs
        removal = input("Do you want to remove the outputs only? (y/n): ").strip().lower()
        delete_folders(removal)
    else:
        print("No data was removed.")

# Run the script
if __name__ == "__main__":
    main()
