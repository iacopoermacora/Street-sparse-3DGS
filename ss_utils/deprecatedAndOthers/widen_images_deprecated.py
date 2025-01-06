import os
from PIL import Image

'''

USELESS CODE AS THE IMAGES HAVE LENS DISTORTION AND SO THE PROJECTION IS DIFFERENT IN ADJACENT IMAGES

'''

def process_images():
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Input and output directories
    input_dir = os.path.join(base_path, "inputs/images")
    output_dir = os.path.join(base_path, "inputs_enlarged/images")
    
    # Camera mappings for adjacent images
    cameras = ["cam1", "cam2", "cam3", "cam4"]
    left_mapping = {"cam1": "cam4", "cam2": "cam1", "cam3": "cam2", "cam4": "cam3"}
    right_mapping = {"cam1": "cam2", "cam2": "cam3", "cam3": "cam4", "cam4": "cam1"}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    for cam in cameras:
        os.makedirs(os.path.join(output_dir, cam), exist_ok=True)

    # Process each camera folder
    for cam in cameras:
        input_folder = os.path.join(input_dir, cam)
        output_folder = os.path.join(output_dir, cam)

        # List all images in the folder
        image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])

        for image_file in image_files:
            # Identify the base number for the current image
            base_number = image_file.split('_')[0]

            # Find matching images in left and right folders
            left_folder = os.path.join(input_dir, left_mapping[cam])
            right_folder = os.path.join(input_dir, right_mapping[cam])

            left_image_file = next((f for f in os.listdir(left_folder) if f.startswith(base_number) and f.endswith('.jpg')), None)
            right_image_file = next((f for f in os.listdir(right_folder) if f.startswith(base_number) and f.endswith('.jpg')), None)

            # Skip if matching files are not found
            if not left_image_file or not right_image_file:
                print(f"Skipping image {image_file}: Missing adjacent images.")
                continue

            # Load the current and adjacent images
            center_image = Image.open(os.path.join(input_folder, image_file))
            left_image = Image.open(os.path.join(left_folder, left_image_file))
            right_image = Image.open(os.path.join(right_folder, right_image_file))

            # Calculate slice width for 15 degrees
            slice_width = int(center_image.width * 0.167)

            # Extract slices
            left_slice = left_image.crop((left_image.width - slice_width, 0, left_image.width, left_image.height))
            right_slice = right_image.crop((0, 0, slice_width, right_image.height))

            # Create new enlarged image
            new_width = center_image.width + 2 * slice_width
            enlarged_image = Image.new('RGB', (new_width, center_image.height))

            # Paste the slices and center image
            enlarged_image.paste(left_slice, (0, 0))
            enlarged_image.paste(center_image, (slice_width, 0))
            enlarged_image.paste(right_slice, (slice_width + center_image.width, 0))

            # Save the enlarged image
            enlarged_image.save(os.path.join(output_folder, image_file))

if __name__ == "__main__":
    process_images()
