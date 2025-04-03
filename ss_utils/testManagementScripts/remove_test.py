import os
import shutil
import argparse

# Function to delete folders
def delete_folders():
    folders_to_delete = ["output"]
    
    # Ask user for confirmation
    for folder in folders_to_delete:
        if os.path.exists(os.path.join(args.project_dir, folder)):
            print(f"Removing folder '{os.path.join(args.project_dir, folder)}")
            # Delete the folder and its contents
            shutil.rmtree(os.path.join(args.project_dir, folder))
            print(f"Folder '{os.path.join(args.project_dir, folder)}' has been removed.")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True, help="Path to the project directory")
    args = parser.parse_args()
    main()
