import numpy as np
from PIL import Image
import os

def visualize_depth_mask(npy_file_path, output_folder):
    """
    Loads a depth mask from a .npy file, normalizes it for visualization,
    and saves it as a PNG image.

    Args:
        npy_file_path (str): Path to the .npy file containing the depth mask.
        output_folder (str): Path to the folder where the PNG image will be saved.
    """
    try:
        # Load the depth mask from the .npy file
        depth_mask = np.load(npy_file_path)

        # Normalize the depth mask to the range 0-255 for visualization
        # You might need to adjust the normalization method depending on your depth data range
        min_depth = np.min(depth_mask)
        max_depth = np.max(depth_mask)

        if max_depth - min_depth != 0:  # Avoid division by zero if all values are the same
            normalized_mask = ((depth_mask - min_depth) / (max_depth - min_depth)) * 255
        else:
            normalized_mask = np.zeros_like(depth_mask) # Or set to a middle gray value like 128 if appropriate

        # Convert to uint8 (8-bit grayscale image)
        normalized_mask = normalized_mask.astype(np.uint8)

        # Create a PIL Image from the numpy array
        img = Image.fromarray(normalized_mask)

        # Construct the output filename (replace .npy with .png)
        filename_base = os.path.splitext(os.path.basename(npy_file_path))[0] # Get filename without .npy extension
        output_filename = os.path.join(output_folder, filename_base + ".png")

        # Save the image as PNG
        img.save(output_filename)
        print(f"Saved visualized depth mask to: {output_filename}")

    except FileNotFoundError:
        print(f"Error: .npy file not found at {npy_file_path}")
    except Exception as e:
        print(f"An error occurred while processing {npy_file_path}: {e}")


def visualize_depth_masks_in_folder(input_folder, output_folder):
    """
    Visualizes all .png.npy files in a given input folder and saves the
    visualized PNG images to the output folder.

    Args:
        input_folder (str): Path to the folder containing .png.npy depth mask files.
        output_folder (str): Path to the folder where visualized PNG images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png.npy"):
            npy_file_path = os.path.join(input_folder, filename)
            visualize_depth_mask(npy_file_path, output_folder)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_depth_folder = f"{current_dir}/visualised_depth_masks"  # Replace with the path to your folder containing .png.npy files
    output_visual_folder = f"{current_dir}/visualised_depth_masks" # Replace with the desired output folder

    # Create example folders if they don't exist (for demonstration purposes)
    if not os.path.exists(input_depth_folder):
        os.makedirs(input_depth_folder)
    if not os.path.exists(output_visual_folder):
        os.makedirs(output_visual_folder)

    # --- Example: Create a dummy .png.npy file for testing ---
    dummy_depth_data = np.random.rand(128, 128) * 1000 # Example depth values range
    dummy_npy_filename = os.path.join(input_depth_folder, "example_depth_mask.png.npy")
    np.save(dummy_npy_filename, dummy_depth_data)
    print(f"Created dummy .npy file: {dummy_npy_filename}")
    # --- End of dummy file creation ---

    visualize_depth_masks_in_folder(input_depth_folder, output_visual_folder)

    print("\nDepth mask visualization process completed.")