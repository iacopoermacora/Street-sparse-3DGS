import os
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import io
import base64
import cv2

# Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Constants
base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
INPUT_DIR = f"{base_path}/ss_utils/static/images"
OUTPUT_DIR = f"{base_path}/inputs/masks"

# List of images and masks
images_path = []
masks_path = []
all_combined_masks = []
all_masks_to_confirm = []
all_masks_to_confirm_name = []

# Load the pre-trained model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()

# Image transforms
transform = T.Compose([T.ToTensor()])

# Helper to get mask predictions with confidence threshold, keeping the original structure
def get_mask(image_path, confidence_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    with torch.no_grad():
        prediction = model([tensor])

    # Find the indices of predictions where score > confidence_threshold
    valid_indices = [idx for idx, score in enumerate(prediction[0]['scores']) if score > confidence_threshold]

    # Filter the prediction object by valid indices
    for key in prediction[0].keys():
        prediction[0][key] = prediction[0][key][valid_indices]

    return prediction

# Save mask as PNG with white background and black mask
def save_mask(mask_array, output_path, image_path):
    # If the mask is empty, create a white image with the same size as the original image
    if mask_array is None:
        print(f'Saving empty mask for {image_path}')
        mask_array = np.zeros((Image.open(image_path).size[1], Image.open(image_path).size[0]), dtype=np.uint8)

    # If there is an additional mask for the face of that image, apply it as well
    # Check face of the image from image_path 
    face = image_path.split('/')[-1].split('.')[0].split('_')[-1]
    print(f'Checking for additional mask for face {face}')
    if os.path.exists(f'{base_path}/ss_utils/manual_masks/manual_mask_{face}.jpg'):
        print('Applying additional mask')
        # Load the additional mask
        additional_mask = cv2.imread(f'{base_path}/ss_utils/manual_masks/manual_mask_{face}.jpg', cv2.IMREAD_GRAYSCALE)

        # Make sure that the mask is binary (0 or 255)
        additional_mask = (additional_mask > 0).astype(np.uint8)

        additional_mask = 1 - additional_mask

        # Count 1 in the additional mask\
        print(f'Number of 1s in the additional mask: {np.count_nonzero(additional_mask)}')

        # Combine the masks (e.g., bitwise OR to keep the union of the two)
        mask_array = np.bitwise_or(mask_array, additional_mask)
    
    mask_array = 1 - mask_array
    
    # Convert to 8-bit image (0-255)
    mask_image = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as PNG, replacing the extension
    output_path = output_path.replace(".jpg", ".png")
    mask_image.save(output_path, format="PNG")

# Detect and process an image
def detect_and_process(image_path, output_path):
    prediction = get_mask(image_path)
    coco_ids_automatic = [1, 16, 18]  # IDs for person and animals (dog, cat)
    coco_ids_confirmation = [2, 3, 4, 6, 7, 8]  # bycicle, car, motorcycle, bus, train, truck

    masks_automatic = []
    masks_to_confirm = []
    person_masks = []
    masks_to_confirm_name = []

    for idx, label in enumerate(prediction[0]['labels']):
        mask = prediction[0]['masks'][idx, 0]
        binary_mask = mask > 0.5  # Assuming mask values range from 0 to 1
        if label.item() == 1:  # Person ID
            person_masks.append(binary_mask)
    for idx, label in enumerate(prediction[0]['labels']):
        mask = prediction[0]['masks'][idx, 0]
        binary_mask = mask > 0.5  # Assuming mask values range from 0 to 1
        if label.item() in coco_ids_automatic:
            masks_automatic.append(mask)
        elif label.item() == 2 or label.item() == 4:  # Bicycle ID
            # Check if the bicycle is in contact with a person
            in_contact = any((binary_mask & person_mask).sum() > 0 for person_mask in person_masks)
            if in_contact:
                masks_to_confirm.append(mask)
                masks_to_confirm_name.append('bicycle/motorbike')
            else:
                print('DISCARDED BICYCLE')
        elif label.item() in coco_ids_confirmation:
            masks_to_confirm.append(mask)
            if label.item() == 3:
                masks_to_confirm_name.append('car')
            elif label.item() == 6:
                masks_to_confirm_name.append('bus')
            elif label.item() == 7:
                masks_to_confirm_name.append('train')
            elif label.item() == 8:
                masks_to_confirm_name.append('truck')
            

    # Combine automatic masks
    combined_mask = np.zeros_like(masks_automatic[0].cpu().numpy(), dtype=np.uint8) if masks_automatic else None
    for mask in masks_automatic:
        combined_mask = np.maximum(combined_mask, (mask.cpu().numpy() > 0.5).astype(np.uint8))

    print(f'Found {len(masks_automatic)} automatic masks and {len(masks_to_confirm)} masks to confirm')

    return combined_mask, masks_to_confirm, masks_to_confirm_name

def superimpose_mask_on_image(image_path, mask_array, opacity=0.3):
    # Load the original image
    original_image = Image.open(image_path).convert("RGBA")

    # Create a red mask with the same size as the original image
    mask = (mask_array > 0.5).astype(np.uint8) * 255  # Binary mask
    mask_image = Image.fromarray(mask, mode="L")

    # Create a red tinted mask with opacity
    red_mask = Image.new("RGBA", original_image.size, (255, 0, 0, int(255 * opacity)))  # Red with 70% opacity
    mask_image_post = Image.composite(red_mask, original_image, mask_image)

    # Blend the original image with the mask
    combined_image = Image.alpha_composite(original_image.convert("RGBA"), mask_image_post)

    # Convert to bytes for rendering
    image_bytes = io.BytesIO()
    combined_image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    # Convert to base64
    return base64.b64encode(image_bytes.getvalue()).decode()

@app.route('/')
def index():
    print('Index page')
    # Initialize the loop index if not already in session
    if 'loop_images' not in session:
        session['loop_index'] = 0
    if 'loop_confirmations' not in session:
        session['loop_confirmations'] = 0
    print("The sessions have been initialized")
    return redirect(url_for('process_item'))

@app.route('/process_item', methods=['GET', 'POST'])
def process_item():
    loop_index = session['loop_index']
    loop_confirmations = session['loop_confirmations']

    if request.method == 'POST':
        # Handle user input and send to another function
        user_choice = request.form['choice']
        print(f'User chose: {user_choice}')  # For now, print the choice
        if user_choice == 'confirm':
            # If the all_combined_masks is empty, instead of adding, create a new mask
            if all_combined_masks[loop_index] is None:
                all_combined_masks[loop_index] = np.zeros_like(all_masks_to_confirm[loop_index][loop_confirmations].cpu().numpy(), dtype=np.uint8)
                all_combined_masks[loop_index] = np.maximum(all_combined_masks[loop_index], (all_masks_to_confirm[loop_index][loop_confirmations].cpu().numpy() > 0.5).astype(np.uint8))
            else:
                all_combined_masks[loop_index] = np.maximum(all_combined_masks[loop_index], (all_masks_to_confirm[loop_index][loop_confirmations].cpu().numpy() > 0.5).astype(np.uint8))
        
        if loop_confirmations >= len(all_masks_to_confirm[loop_index]) - 1:
            # Update the output path to use .png extension
            output_path = masks_path[loop_index].replace('.jpg', '.png')
            save_mask(all_combined_masks[loop_index], output_path, images_path[loop_index])
            # Move to the next image
            session['loop_index'] += 1
            session['loop_confirmations'] = 0
            return redirect(url_for('process_item'))
        # Move to the next item in the loop
        session['loop_confirmations'] += 1
        return redirect(url_for('process_item'))

    print(f'Loop index: {loop_index}, Loop confirmations: {loop_confirmations}')
    
    if loop_index >= len(images_path):  # If all items have been processed
        return redirect(url_for('final_page'))
    
    if loop_confirmations == 0:
        print(f'Processing image {loop_index}')
        # Update the check for existing mask to use .png extension
        png_mask_path = masks_path[loop_index].replace('.jpg', '.png')
        if os.path.exists(png_mask_path):
            print(f'Skipping image {loop_index} ({png_mask_path}) as mask already exists')
            all_combined_masks.append(None)
            all_masks_to_confirm.append(None)
            all_masks_to_confirm_name.append(None)
            session['loop_index'] += 1
            session['loop_confirmations'] = 0
            return redirect(url_for('process_item'))
        combined_masks, masks_to_confirm, masks_to_confirm_name = detect_and_process(images_path[loop_index], png_mask_path)
        print(f'There are {len(masks_to_confirm)} masks to confirm')
        all_combined_masks.append(combined_masks)
        all_masks_to_confirm.append(masks_to_confirm)
        all_masks_to_confirm_name.append(masks_to_confirm_name)
    
    # If there are masks to confirm
    if loop_confirmations < len(all_masks_to_confirm[loop_index]):
        print(f'Asking for confirmation for image {loop_index}')
        # Keep just the filename and the last foldername using split and join using '/'
        current_image = images_path[loop_index].split('/')[-2] + '/' + images_path[loop_index].split('/')[-1]
        current_mask = all_masks_to_confirm[loop_index][loop_confirmations].cpu().numpy()

        # Superimpose the mask on the image
        mask_base64 = superimpose_mask_on_image(images_path[loop_index], current_mask, opacity=0.7)
    else:
        print(f'Saving mask for image {loop_index} as there are no masks to confirm')
        # Update the output path to use .png extension
        output_path = masks_path[loop_index].replace('.jpg', '.png')
        save_mask(all_combined_masks[loop_index], output_path, images_path[loop_index])
        # Move to the next item in the loop
        session['loop_index'] += 1
        session['loop_confirmations'] = 0
        return redirect(url_for('process_item'))
    
    print('Rendering the confirm page')
    return render_template('confirm.html', image=current_image, mask="data:image/png;base64," + mask_base64, mask_to_confirm_name=all_masks_to_confirm_name[loop_index][loop_confirmations])

@app.route('/final_page')
def final_page():
    return render_template('final_page.html')

# Serve images from the inputs folder
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('/host/inputs/images', filename)

# Process all images
def process_all_images():
    print('I am processing all images')
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".jpg"):
                print(f'Processing {file}')
                image_path = os.path.join(root, file)
                mask_path = image_path.replace(INPUT_DIR, OUTPUT_DIR)
                images_path.append(image_path)
                masks_path.append(mask_path)

# Main entry point
if __name__ == '__main__':
    process_all_images()
    app.run(host="0.0.0.0", port=5000)
