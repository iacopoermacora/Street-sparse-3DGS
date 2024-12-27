import os
import shutil

# Function to delete folders
def delete_folders():
    folders_to_delete = ["camera_calibration", "output"]

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

# Main function to run the script
def main():
    # Prompt the user for confirmation to delete the folders
    confirmation = input("Do you want to remove the current test data? Make sure to have saved it before proceeding. (y/n): ").strip().lower()
    
    if confirmation == 'y':
        delete_folders()
    else:
        print("No data was removed.")

# Run the script
if __name__ == "__main__":
    main()
