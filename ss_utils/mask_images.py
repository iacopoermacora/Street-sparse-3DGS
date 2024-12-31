import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import os

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Class indices for Mask R-CNN (COCO Dataset)
COCO_CLASSES = {
    1: 'person',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
}

# Function to process image and apply masks
def apply_rmaskcnn(image_path, threshold=0.5):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Transform image for the model
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Prepare masks
    final_mask = np.zeros((height, width), dtype=np.uint8)

    # Define classes to mask
    people_and_animals = {1, 17, 18, 19}
    vehicles = {2, 3, 4}

    for i, label in enumerate(predictions['labels'].numpy()):
        score = predictions['scores'][i].item()
        if score < threshold:
            continue  # Ignore detections with low confidence

        mask = predictions['masks'][i, 0].cpu().numpy() > 0.5
        mask = mask.astype(np.uint8)

        if label in people_and_animals or label in vehicles:
            final_mask = cv2.bitwise_or(final_mask, mask)

    return final_mask, image

# Example usage
image_path = '/host/inputs/images/cam1/0015_WE931VUC_f1.jpg'
mask, original_image = apply_rmaskcnn(image_path)

# Create the output directory if it doesn't exist
output_path = '/host/inputs/masks/cam1'
os.makedirs(output_path, exist_ok=True)

# Visualize the mask
overlay = cv2.addWeighted(original_image, 0.7, cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), 0.3, 0)

# Save the mask and overlay images
mask_output_path = os.path.join(output_path, 'mask.jpg')
overlay_output_path = os.path.join(output_path, 'overlay.jpg')

cv2.imwrite(mask_output_path, mask * 255)  # Saving the mask (multiplied by 255 to convert it from 0-1 to 0-255)
cv2.imwrite(overlay_output_path, overlay)  # Saving the overlay image

print(f"Mask saved to {mask_output_path}")
print(f"Overlay saved to {overlay_output_path}")

