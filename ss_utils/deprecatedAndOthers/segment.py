import os
import glob
from typing import Tuple, Optional, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
import time
# tqdm is removed as it was used for video progress

# Assuming utils.video.create_directory is available or reimplemented if needed
# For simplicity, we'll use os.makedirs
# from utils.video import create_directory

from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_OPEN_VOCABULARY_DETECTION_TASK, FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, \
    FLORENCE_DENSE_REGION_CAPTION_TASK # Added for region-to-category validation
from utils.modes import IMAGE_OPEN_VOCABULARY_DETECTION_MODE #
from utils.sam import load_sam_image_model, run_sam_inference #
import supervision as sv

# --- Configuration ---
INPUT_FOLDER_BASE = "camera_calibration/rectified/images"  # Base folder for input images
TEST_FILE = "camera_calibration/extras/test.txt"  # File containing image paths relative to INPUT_FOLDER_BASE
OUTPUT_ANNOTATED_FOLDER = "camera_calibration/rectified/semantic_segmentation/masks_annotated"  # Folder to save annotated images (with background and boxes)
OUTPUT_MASKS_FOLDER = "camera_calibration/rectified/semantic_segmentation/masks"  # Folder to save mask-only images
FIXED_TEXT_PROMPT = "sky, ground, floor, road, house, car, bike, plant, vegetation, lampost" # CHANGE THIS to your desired comma-separated prompts

# Define category groups with consistent colors
CATEGORY_GROUPS = {
    "sky": {"color": "#87CEEB", "categories": ["sky"]},  # Sky Blue
    "ground": {"color": "#8B4513", "categories": ["ground", "floor", "road"]},  # Saddle Brown
    "structure": {"color": "#696969", "categories": ["house", "building", "wall"]},  # Dim Gray
    "vehicle": {"color": "#FF4500", "categories": ["car", "bike"]},  # Orange Red
    "vegetation": {"color": "#228B22", "categories": ["vegetation", "plant"]},  # Forest Green
    "lighting": {"color": "#FFD700", "categories": ["lampost"]}  # Gold
}

# Create a mapping from category to color
CATEGORY_TO_COLOR = {}
CATEGORY_TO_GROUP = {}
for group_name, group_info in CATEGORY_GROUPS.items():
    for category in group_info["categories"]:
        CATEGORY_TO_COLOR[category] = group_info["color"]
        CATEGORY_TO_GROUP[category] = group_name

# Define category validation mapping
CATEGORY_VALIDATION_MAP = {
    "sky": ["sky", "clouds", "cloud", "air", "atmosphere", "heaven", "blue sky", "cloudy sky"],
    "ground": ["ground", "earth", "soil", "dirt", "floor", "surface", "terrain", "land", "pavement", "road", "sidewalk"],
    "floor": ["ground", "earth", "soil", "dirt", "floor", "surface", "terrain", "land", "pavement", "road", "sidewalk"],
    "road": ["ground", "earth", "soil", "dirt", "floor", "surface", "terrain", "land", "pavement", "road", "sidewalk"],
    "house": ["house", "building", "home", "structure", "residence", "dwelling", "architecture", "construction"],
    "building": ["building", "structure", "house", "edifice", "construction", "architecture", "facility"],
    "wall": ["wall", "fence", "partition", "barrier", "enclosure", "boundary", "structure"],
    "car": ["car", "vehicle", "automobile", "sedan", "suv", "truck", "transportation", "motor vehicle"],
    "bike": ["bike", "bicycle", "motorcycle", "motorbike", "cycle", "two-wheeler", "scooter"],
    "plant": ["plant", "vegetation", "tree", "bush", "flower", "leaf", "greenery", "flora", "garden", "nature"],
    "vegetation": ["vegetation", "plant", "greenery", "tree", "bush", "grass", "foliage", "flora", "nature", "garden"],
    'lampost': ['lamp', 'streetlight', 'light', 'lamp post', 'street lamp', 'pole', 'lighting']
}

# Define priority hierarchy for overlapping masks (higher number = higher priority)
CATEGORY_PRIORITY = {
    "sky": 1,
    "ground": 2,
    "floor": 2,
    "road": 2,
    "house": 3,
    "building": 3,
    "wall": 3,
    "car": 3,
    "bike": 3,
    "vegetation": 3,
    "plant": 3,
    "lampost": 3,
    "traffic sign": 3
}

def create_output_directories():
    """Create output directories maintaining the same structure as input"""
    if not os.path.exists(OUTPUT_ANNOTATED_FOLDER):
        os.makedirs(OUTPUT_ANNOTATED_FOLDER)
    if not os.path.exists(OUTPUT_MASKS_FOLDER):
        os.makedirs(OUTPUT_MASKS_FOLDER)

def read_test_file(test_file_path: str) -> List[str]:
    """Read image paths from test.txt file"""
    if not os.path.exists(test_file_path):
        print(f"Test file {test_file_path} not found!")
        return []
    
    image_paths = []
    with open(test_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                # Construct full path
                full_path = os.path.join(INPUT_FOLDER_BASE, line)
                if os.path.exists(full_path):
                    image_paths.append((line, full_path))  # (relative_path, full_path)
                else:
                    print(f"Warning: Image not found: {full_path}")
    
    return image_paths

def get_output_path(relative_path: str, output_base: str) -> str:
    """Create output path maintaining directory structure"""
    output_path = os.path.join(output_base, relative_path)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_path

# --- Model and Annotator Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Enter autocast context manager once if possible, or manage per function
# torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu", dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32).__enter__()
# The original script uses cuda specific autocast. Let's refine this for broader compatibility or ensure it's handled within processing.
if DEVICE.type == "cuda":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8: #
        torch.backends.cuda.matmul.allow_tf32 = True #
        torch.backends.cudnn.allow_tf32 = True #

print("Loading Florence model...")
FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE) #
print("Loading SAM Image model...")
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE) #

# Create consistent color palette based on groups
GROUP_COLORS = [group_info["color"] for group_info in CATEGORY_GROUPS.values()]
COLOR_PALETTE = sv.ColorPalette.from_hex(GROUP_COLORS)

# We'll create annotators dynamically with proper color assignment

def get_color_index_for_category(category: str) -> int:
    """Get consistent color index for a category based on its group"""
    group = CATEGORY_TO_GROUP.get(category.lower())
    if group:
        group_names = list(CATEGORY_GROUPS.keys())
        return group_names.index(group)
    return 0  # Default to first color if not found

def assign_consistent_colors_to_detections(detections_list: List[sv.Detections], category_names: List[str]) -> Tuple[sv.Detections, List[str]]:
    """Assign consistent colors to detections based on category groups and return merged detections with category labels"""
    if not detections_list:
        return sv.Detections.empty(), []
    
    all_xyxy = []
    all_masks = []
    all_confidences = []
    all_class_ids = []
    all_labels = []
    
    for detections, category in zip(detections_list, category_names):
        if len(detections.xyxy) > 0:
            # Get consistent color index for this category
            color_index = get_color_index_for_category(category)
            
            # Collect data for merging
            all_xyxy.extend(detections.xyxy)
            if detections.mask is not None:
                all_masks.extend(detections.mask)
            
            # Handle confidence scores
            if detections.confidence is not None:
                all_confidences.extend(detections.confidence)
            else:
                all_confidences.extend([1.0] * len(detections.xyxy))
            
            # Assign consistent class_id and labels
            all_class_ids.extend([color_index] * len(detections.xyxy))
            all_labels.extend([category] * len(detections.xyxy))
    
    if not all_xyxy:
        return sv.Detections.empty(), []
    
    # Create merged detections with consistent class_ids
    merged_detections = sv.Detections(
        xyxy=np.array(all_xyxy),
        mask=np.array(all_masks) if all_masks else None,
        confidence=np.array(all_confidences),
        class_id=np.array(all_class_ids, dtype=int)
    )
    
    return merged_detections, all_labels

def annotate_image_pil(image_pil: Image.Image, detections: sv.Detections, labels: List[str]) -> Image.Image:
    """ Annotates a PIL image with detections and returns a PIL image. """
    image_np = np.array(image_pil.convert("RGB"))
    output_image_np = image_np.copy()
    
    # Create annotators with consistent colors
    mask_annotator = sv.MaskAnnotator(
        color=COLOR_PALETTE, 
        color_lookup=sv.ColorLookup.CLASS
    )
    box_annotator = sv.BoxAnnotator(
        color=COLOR_PALETTE, 
        color_lookup=sv.ColorLookup.CLASS
    )
    label_annotator = sv.LabelAnnotator(
        color=COLOR_PALETTE,
        color_lookup=sv.ColorLookup.CLASS,
        text_position=sv.Position.CENTER_OF_MASS,
        text_color=sv.Color.from_hex("#000000"),
        border_radius=5
    )
    
    output_image_np = mask_annotator.annotate(output_image_np, detections)
    output_image_np = box_annotator.annotate(output_image_np, detections)
    output_image_np = label_annotator.annotate(output_image_np, detections, labels=labels)
    
    return Image.fromarray(output_image_np)

def create_mask_only_image(image_pil: Image.Image, detections: sv.Detections) -> Image.Image:
    """Create an image with only masks (no background, no bounding boxes) at maximum opacity"""
    # Create a black background
    image_np = np.zeros((image_pil.height, image_pil.width, 3), dtype=np.uint8)
    
    # Create mask annotator with full opacity and consistent colors
    mask_annotator = sv.MaskAnnotator(
        color=COLOR_PALETTE,
        color_lookup=sv.ColorLookup.CLASS,
        opacity=1.0  # Maximum opacity
    )
    
    # Apply only masks
    output_image_np = mask_annotator.annotate(image_np, detections)
    return Image.fromarray(output_image_np)

def calculate_mask_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate the overlap ratio between two masks.
    Returns the ratio of overlapping area to the smaller mask's area.
    """
    if mask1 is None or mask2 is None:
        return 0.0
    
    # Ensure masks are boolean
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    
    # Calculate intersection and areas
    intersection = np.logical_and(mask1_bool, mask2_bool)
    intersection_area = np.sum(intersection)
    
    if intersection_area == 0:
        return 0.0
    
    # Calculate overlap ratio relative to smaller mask
    mask1_area = np.sum(mask1_bool)
    mask2_area = np.sum(mask2_bool)
    smaller_area = min(mask1_area, mask2_area)
    
    if smaller_area == 0:
        return 0.0
    
    return intersection_area / smaller_area


def resolve_overlapping_masks(
    detections_list: List[sv.Detections], 
    category_names: List[str],
    overlap_threshold: float = 0.8
) -> List[sv.Detections]:
    """
    Resolves overlapping masks by keeping higher priority categories and removing lower priority ones.
    
    Args:
        detections_list: List of detections for each category
        category_names: Corresponding category names for each detection set
        overlap_threshold: Minimum overlap ratio to consider masks as overlapping
    
    Returns:
        List of filtered detections with overlapping masks resolved
    """
    if not detections_list or len(detections_list) != len(category_names):
        return detections_list
    
    # Create a flat list of all detections with their category info
    all_detections = []
    for i, (detections, category) in enumerate(zip(detections_list, category_names)):
        if detections.mask is not None and len(detections.mask) > 0:
            for j in range(len(detections.mask)):
                all_detections.append({
                    'category_idx': i,
                    'detection_idx': j,
                    'category': category,
                    'priority': CATEGORY_PRIORITY.get(category.lower(), 0),
                    'mask': detections.mask[j],
                    'bbox': detections.xyxy[j],
                    'detections_obj': detections
                })
    
    if len(all_detections) < 2:
        return detections_list
    
    # Find overlapping pairs and resolve conflicts
    to_remove = set()  # Store (category_idx, detection_idx) tuples to remove
    
    print(f"Checking {len(all_detections)} detections for overlaps...")
    
    for i in range(len(all_detections)):
        for j in range(i + 1, len(all_detections)):
            det1 = all_detections[i]
            det2 = all_detections[j]
            
            # Skip if either detection is already marked for removal
            if (det1['category_idx'], det1['detection_idx']) in to_remove or \
               (det2['category_idx'], det2['detection_idx']) in to_remove:
                continue
            
            # Calculate overlap
            overlap_ratio = calculate_mask_overlap(det1['mask'], det2['mask'])
            
            if overlap_ratio > overlap_threshold:
                print(f"Found overlap ({overlap_ratio:.2f}) between {det1['category']} and {det2['category']}")
                
                # Remove the lower priority detection
                if det1['priority'] > det2['priority']:
                    to_remove.add((det2['category_idx'], det2['detection_idx']))
                    print(f"Removing {det2['category']} (priority {det2['priority']}) in favor of {det1['category']} (priority {det1['priority']})")
                elif det2['priority'] > det1['priority']:
                    to_remove.add((det1['category_idx'], det1['detection_idx']))
                    print(f"Removing {det1['category']} (priority {det1['priority']}) in favor of {det2['category']} (priority {det2['priority']})")
                else:
                    # Same priority, remove the one with smaller area
                    area1 = np.sum(det1['mask'])
                    area2 = np.sum(det2['mask'])
                    if area1 < area2:
                        to_remove.add((det1['category_idx'], det1['detection_idx']))
                        print(f"Removing smaller {det1['category']} in favor of larger {det2['category']}")
                    else:
                        to_remove.add((det2['category_idx'], det2['detection_idx']))
                        print(f"Removing smaller {det2['category']} in favor of larger {det1['category']}")
    
    # Create filtered detection lists
    filtered_detections_list = []
    for i, detections in enumerate(detections_list):
        if detections.mask is None or len(detections.mask) == 0:
            filtered_detections_list.append(detections)
            continue
        
        # Find indices to keep for this category
        indices_to_remove = [idx for cat_idx, idx in to_remove if cat_idx == i]
        indices_to_keep = [j for j in range(len(detections.mask)) if j not in indices_to_remove]
        
        if not indices_to_keep:
            # All detections removed, return empty
            filtered_detections_list.append(sv.Detections.empty())
        elif len(indices_to_keep) == len(detections.mask):
            # No detections removed, keep original
            filtered_detections_list.append(detections)
        else:
            # Filter the detections
            keep_mask = np.zeros(len(detections.mask), dtype=bool)
            keep_mask[indices_to_keep] = True
            filtered_detections_list.append(detections[keep_mask])
    
    print(f"Overlap resolution complete. Removed {len(to_remove)} conflicting detections.")
    return filtered_detections_list


def crop_image_from_bbox(image_pil: Image.Image, bbox: List[float]) -> Image.Image:
    """
    Crops the image using the bounding box coordinates.
    bbox format: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    return image_pil.crop((int(x1), int(y1), int(x2), int(y2)))


def validate_detection_with_region_caption(
    image_pil: Image.Image, 
    bbox: List[float], 
    expected_category: str,
    confidence_threshold: float = 0.3
) -> bool:
    """
    Validates a detection by cropping the region and using Florence2's region captioning
    to check if the detected object matches the expected category.
    """
    try:
        # Crop the region
        cropped_image = crop_image_from_bbox(image_pil, bbox)
        
        # Use Florence2 to generate a detailed caption of the cropped region
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=cropped_image,
            task=FLORENCE_DETAILED_CAPTION_TASK,
            text=""
        )
        
        # Extract the caption
        caption = result.get(FLORENCE_DETAILED_CAPTION_TASK, "").lower()
        
        if not caption:
            print(f"No caption generated for region, rejecting detection")
            return False
        
        # Check if any of the valid categories for this expected category appear in the caption
        valid_categories = CATEGORY_VALIDATION_MAP.get(expected_category.lower(), [expected_category.lower()])
        
        # Count matches
        matches = sum(1 for category in valid_categories if category.lower() in caption)
        
        is_valid = matches > 0
        
        if is_valid:
            print(f"Valid detection for '{expected_category}': Valid={is_valid}'")
        else:
            print(f"Validation for '{expected_category}': Caption='{caption}', Valid={is_valid}")
        
        return is_valid
        
    except Exception as e:
        print(f"Error validating detection: {e}")
        return False


def filter_valid_detections(
    image_pil: Image.Image, 
    detections: sv.Detections, 
    expected_category: str
) -> sv.Detections:
    """
    Filters detections by validating each bounding box using region captioning.
    """
    if len(detections.xyxy) == 0:
        return detections
    
    # Create a boolean mask for valid detections
    valid_mask = np.zeros(len(detections.xyxy), dtype=bool)
    
    for i, bbox in enumerate(detections.xyxy):
        if validate_detection_with_region_caption(image_pil, bbox, expected_category):
            valid_mask[i] = True
        else:
            print(f"Rejected detection {i} for category '{expected_category}'")
    
    if not np.any(valid_mask):
        # Return empty detections
        return sv.Detections.empty()
    
    # Use the boolean mask to filter detections
    # This is a safer approach that handles all data properly
    return detections[valid_mask]


@torch.inference_mode() #
def process_single_image(
    image_pil: Image.Image, text_prompt: str
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Processes a single image for open vocabulary detection and segmentation.
    Returns (annotated_image, mask_only_image)
    """
    # Determine if operating on CUDA or CPU and set autocast accordingly
    use_cuda = DEVICE.type == "cuda"
    autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16} if use_cuda else {"device_type": "cpu", "dtype": torch.float32}

    with torch.autocast(**autocast_kwargs): # Simplified autocast handling
        texts = [prompt.strip() for prompt in text_prompt.split(",")]
        initial_detections_list = []
        initial_category_names = [] # Store category names corresponding to initial_detections_list

        for text in texts:
            print(f"Processing category: {text}")
            _, result = run_florence_inference(
                model=FLORENCE_MODEL,
                processor=FLORENCE_PROCESSOR,
                device=DEVICE,
                image=image_pil,
                task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                text=text
            )
            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2,
                result=result,
                resolution_wh=image_pil.size
            )

            if len(detections.xyxy) > 0:
                print(f"Found {len(detections.xyxy)} initial detections for '{text}', validating...")
                detections = filter_valid_detections(image_pil, detections, text)
                print(f"After validation: {len(detections.xyxy)} valid detections for '{text}'")

                if len(detections.xyxy) > 0:
                    detections = run_sam_inference(SAM_IMAGE_MODEL, image_pil, detections)
                    if detections.mask is not None and len(detections.mask) > 0: # Ensure SAM actually produced masks
                        initial_detections_list.append(detections)
                        initial_category_names.append(text) # Add category name
                    else:
                        print(f"No masks after SAM for '{text}', skipping this category's detections.")
                else:
                    print(f"No valid detections after validation for '{text}'")
            else:
                print(f"No initial detections found for '{text}'")

        if not initial_detections_list:
            print(f"No valid detections found for any prompt in the image after initial processing.")
            return None, None

        # At this point, initial_detections_list and initial_category_names are aligned.
        # Each element in initial_detections_list corresponds to the category name
        # at the same index in initial_category_names.

        # Resolve overlapping masks
        # resolve_overlapping_masks should ideally return a list of detections
        # where each element still corresponds to initial_category_names,
        # but some detections within those sv.Detections objects might be filtered out.
        # If an sv.Detections object becomes entirely empty, we'll handle it next.
        if len(initial_detections_list) > 1:
            print("Resolving overlapping masks...")
            # Pass the aligned lists
            resolved_detections_list = resolve_overlapping_masks(initial_detections_list, initial_category_names)
        else:
            resolved_detections_list = initial_detections_list

        # Filter out categories for which all detections were removed by overlap resolution
        # or if they were empty to begin with after SAM.
        # We need to build the final lists for merging that are perfectly aligned.
        final_detections_to_merge = []
        final_category_names_for_merge = []

        for i, dets in enumerate(resolved_detections_list):
            # initial_category_names[i] is the category for resolved_detections_list[i]
            if dets.mask is not None and len(dets.mask) > 0: # Check if there are any masks left
                final_detections_to_merge.append(dets)
                final_category_names_for_merge.append(initial_category_names[i]) # Keep the corresponding category name

        if not final_detections_to_merge:
            print(f"No valid detections remaining after overlap resolution and filtering.")
            return None, None

        # Now, final_detections_to_merge and final_category_names_for_merge are aligned.
        merged_detections, all_labels = assign_consistent_colors_to_detections(
            final_detections_to_merge,
            final_category_names_for_merge # Use the correctly filtered and aligned category names
        )

        if len(merged_detections.xyxy) > 0 and merged_detections.mask is not None and len(merged_detections.mask) > 0:
            # Optional: A final SAM pass on the merged detections if it helps refine combined masks.
            # Consider if this is truly necessary or if the masks from individual SAM runs are sufficient.
            # If run, ensure class_id and other attributes are preserved or correctly reassigned.
            # For now, let's assume previous SAM masks are good enough after merging.
            # merged_detections = run_sam_inference(SAM_IMAGE_MODEL, image_pil, merged_detections)
            pass # Assuming SAM was run per category and masks are fine. Re-running SAM here might lose original category associations if not handled carefully.
        else: # Handle case where merging might result in no usable detections
             if not (merged_detections.mask is not None and len(merged_detections.mask) > 0):
                print(f"No objects with masks found after merging for prompt '{text_prompt}' in the image.")
                return None, None


        if len(merged_detections.xyxy) == 0 and (merged_detections.mask is None or merged_detections.mask.size == 0):
            print(f"No objects found for prompt '{text_prompt}' in the image after all processing.")
            return None, None

        annotated_image = annotate_image_pil(image_pil, merged_detections, all_labels)
        mask_only_image = create_mask_only_image(image_pil, merged_detections)

        return annotated_image, mask_only_image

def main():
    start_time = time.time()

    # Create output directories
    create_output_directories()

    # Read image paths from test.txt
    print(f"Reading image paths from: {TEST_FILE}")
    image_paths_data = read_test_file(TEST_FILE)
    
    if not image_paths_data:
        print(f"No valid images found in {TEST_FILE}. Please check the file and paths.")
        return

    print(f"Found {len(image_paths_data)} images to process.")

    for relative_path, full_path in image_paths_data:
        print(f"\nProcessing image: {full_path}")
        try:
            image_pil = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {full_path}: {e}")
            continue

        annotated_image, mask_only_image = process_single_image(image_pil, FIXED_TEXT_PROMPT)

        if annotated_image and mask_only_image:
            try:
                # Save annotated image (with background and bounding boxes)
                annotated_output_path = get_output_path(relative_path, OUTPUT_ANNOTATED_FOLDER)
                annotated_image.save(annotated_output_path)
                print(f"Saved annotated image to: {annotated_output_path}")
                
                # Save mask-only image
                mask_output_path = get_output_path(relative_path, OUTPUT_MASKS_FOLDER)
                mask_only_image.save(mask_output_path)
                print(f"Saved mask-only image to: {mask_output_path}")
                
            except Exception as e:
                print(f"Error saving images for {relative_path}: {e}")
        else:
            print(f"Skipping save for {relative_path} as no annotations were made or an error occurred.")

    print("\nProcessing complete.")
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()