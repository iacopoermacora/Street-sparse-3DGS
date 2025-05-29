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
    "car": 3,
    "bike": 3,
    "vegetation": 3,
    "plant": 3,
    "lampost": 3,
    "traffic sign": 3,
    "house": 4,  # Highest priority
    "building": 4,
    "wall": 4
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

if DEVICE.type == "cuda":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8: #
        torch.backends.cuda.matmul.allow_tf32 = True #
        torch.backends.cudnn.allow_tf32 = True #

print("Loading Florence model...")
FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE) #
print("Loading SAM Image model...")
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE) #

GROUP_COLORS = [group_info["color"] for group_info in CATEGORY_GROUPS.values()]
COLOR_PALETTE = sv.ColorPalette.from_hex(GROUP_COLORS)

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
            color_index = get_color_index_for_category(category)
            
            all_xyxy.extend(detections.xyxy)
            if detections.mask is not None:
                all_masks.extend(detections.mask)
            
            if detections.confidence is not None:
                all_confidences.extend(detections.confidence)
            else:
                all_confidences.extend([1.0] * len(detections.xyxy))
            
            all_class_ids.extend([color_index] * len(detections.xyxy))
            all_labels.extend([category] * len(detections.xyxy))
    
    if not all_xyxy:
        return sv.Detections.empty(), []
    
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
    image_np = np.zeros((image_pil.height, image_pil.width, 3), dtype=np.uint8)
    
    mask_annotator = sv.MaskAnnotator(
        color=COLOR_PALETTE,
        color_lookup=sv.ColorLookup.CLASS,
        opacity=1.0  # Maximum opacity
    )
    
    output_image_np = mask_annotator.annotate(image_np, detections)
    return Image.fromarray(output_image_np)

def calculate_mask_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate the overlap ratio between two masks.
    Returns the ratio of overlapping area to the smaller mask's area.
    """
    if mask1 is None or mask2 is None or mask1.shape != mask2.shape:
        return 0.0
    
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    
    intersection = np.logical_and(mask1_bool, mask2_bool)
    intersection_area = np.sum(intersection)
    
    if intersection_area == 0:
        return 0.0
    
    mask1_area = np.sum(mask1_bool)
    mask2_area = np.sum(mask2_bool)
    
    # Avoid division by zero if a mask is empty after boolean conversion (e.g. all False)
    if mask1_area == 0 and mask2_area == 0 : return 0.0
    
    smaller_area = min(mask1_area, mask2_area)
    if smaller_area == 0: # If the smaller mask has no true pixels, overlap effectively becomes 0 or undefined.
                          # but if intersection_area > 0, this means smaller_area was originally > 0.
                          # This case should be handled by intersection_area == 0 if smaller_area is truly 0.
                          # If smaller_area is 0 but intersection_area > 0, it's a contradiction unless areas are calculated before intersection.
                          # Re-evaluating: if smaller_area is 0, overlap calculation is problematic.
                          # However, if intersection_area > 0, then smaller_area must have been > 0.
                          # The only case for smaller_area == 0 here is if one mask is all False.
        return 0.0 # Or handle as an error, but 0.0 is safe for ratio.

    return intersection_area / smaller_area


def resolve_overlapping_masks(
    detections_list: List[sv.Detections], 
    category_names: List[str],
    overlap_threshold: float = 0.8
) -> List[sv.Detections]:
    """
    Resolves overlapping masks based on priority, size, and coverage.
    Modifies masks in detections_list directly for the "on-top" scenario.
    
    Args:
        detections_list: List of detections for each category. Masks may be modified in-place.
        category_names: Corresponding category names for each detection set.
        overlap_threshold: Minimum overlap ratio (intersection/smaller_area) to consider masks as overlapping.
    
    Returns:
        List of filtered sv.Detections objects with overlapping masks resolved.
    """
    if not detections_list or len(detections_list) != len(category_names):
        return detections_list
    
    # Define thresholds for new logic
    SIZE_SIMILARITY_THRESHOLD = 0.2  # Masks are similar if areas are within 20% of each other (relative to smaller)
    COMPLETE_COVERAGE_THRESHOLD = 0.95 # Smaller mask is "completely covered" if overlap ratio > 0.95
    WAY_BIGGER_THRESHOLD = 5.0       # Larger mask is "way bigger" if its area is > 5x smaller mask's area

    all_detections = []
    for i, (detections, category) in enumerate(zip(detections_list, category_names)):
        if detections.mask is not None and len(detections.mask) > 0:
            for j in range(len(detections.mask)):
                all_detections.append({
                    'category_idx': i,
                    'detection_idx': j,
                    'category': category,
                    'priority': CATEGORY_PRIORITY.get(category.lower(), 0),
                    'mask': detections.mask[j], # This is a reference to the mask array
                    'bbox': detections.xyxy[j],
                    # 'detections_obj': detections # Not strictly needed if we update detections_list directly
                })
    
    if len(all_detections) < 2:
        return detections_list
    
    to_remove = set()  # Store (category_idx, detection_idx) tuples to remove
    
    print(f"Checking {len(all_detections)} detections for overlaps...")
    
    for i in range(len(all_detections)):
        for j in range(i + 1, len(all_detections)):
            det1_info = all_detections[i]
            det2_info = all_detections[j]
            
            # Skip if either detection is already marked for removal or if they are the same detection somehow
            if (det1_info['category_idx'], det1_info['detection_idx']) in to_remove or \
               (det2_info['category_idx'], det2_info['detection_idx']) in to_remove:
                continue
            
            # Get current masks (could have been modified by a previous iteration for Condition 2)
            mask1 = det1_info['mask']
            mask2 = det2_info['mask']

            if mask1 is None or mask2 is None: # Should not happen if populated correctly
                continue

            overlap_ratio = calculate_mask_overlap(mask1, mask2)
            
            if overlap_ratio > overlap_threshold:
                print(f"\nFound overlap ({overlap_ratio:.2f}) between: \n"
                      f"  Det1: {det1_info['category']} (cat_idx {det1_info['category_idx']}, det_idx {det1_info['detection_idx']}) \n"
                      f"  Det2: {det2_info['category']} (cat_idx {det2_info['category_idx']}, det_idx {det2_info['detection_idx']})")

                area1 = np.sum(mask1.astype(bool))
                area2 = np.sum(mask2.astype(bool))

                # --- Condition 1: Overlap, similar size, different priority ---
                are_sizes_similar = False
                if min(area1, area2) > 0: # Avoid division by zero and meaningless comparison for empty masks
                    if abs(area1 - area2) / min(area1, area2) < SIZE_SIMILARITY_THRESHOLD:
                        are_sizes_similar = True
                
                if are_sizes_similar and det1_info['priority'] != det2_info['priority']:
                    if det1_info['priority'] > det2_info['priority']:
                        to_remove.add((det2_info['category_idx'], det2_info['detection_idx']))
                        print(f"  Action (Cond 1): Removing Det2 ({det2_info['category']}) - similar size, lower priority ({det2_info['priority']}) than Det1 ({det1_info['priority']}).")
                    else:
                        to_remove.add((det1_info['category_idx'], det1_info['detection_idx']))
                        print(f"  Action (Cond 1): Removing Det1 ({det1_info['category']}) - similar size, lower priority ({det1_info['priority']}) than Det2 ({det2_info['priority']}).")
                    continue # Conflict resolved for this pair

                # --- Condition 2: Overlap, one completely covered, other way bigger ---
                if area1 > 0 and area2 > 0: # Both masks must have an area to be considered for this condition
                    smaller_det_info, larger_det_info = (det1_info, det2_info) if area1 < area2 else (det2_info, det1_info)
                    smaller_area = min(area1, area2)
                    larger_area = max(area1, area2)
                    
                    # overlap_ratio is intersection_area / smaller_mask_area.
                    is_smaller_completely_covered = overlap_ratio > COMPLETE_COVERAGE_THRESHOLD
                    is_larger_way_bigger = (larger_area / smaller_area) > WAY_BIGGER_THRESHOLD

                    if is_smaller_completely_covered and is_larger_way_bigger:
                        print(f"  Action (Cond 2): Keeping both. {smaller_det_info['category']} (smaller) is mostly covered by overlap, "
                              f"and {larger_det_info['category']} (larger) is much bigger.")
                        print(f"                   Smaller mask ({smaller_det_info['category']}) will be 'on top'. Modifying larger mask ({larger_det_info['category']}).")

                        # Modify the larger mask: larger_mask = larger_mask - smaller_mask
                        # Ensure masks are boolean for the operation
                        smaller_mask_bool = smaller_det_info['mask'].astype(bool)
                        larger_mask_bool = larger_det_info['mask'].astype(bool)
                        
                        # Subtract the smaller mask from the larger mask's area
                        updated_larger_mask_arr = np.logical_and(larger_mask_bool, np.logical_not(smaller_mask_bool))
                        
                        # Get original dtype of the larger mask to preserve it
                        original_larger_mask_dtype = detections_list[larger_det_info['category_idx']].mask[larger_det_info['detection_idx']].dtype
                        
                        # Update the mask in the original sv.Detections object in detections_list
                        detections_list[larger_det_info['category_idx']].mask[larger_det_info['detection_idx']] = updated_larger_mask_arr.astype(original_larger_mask_dtype)
                        
                        # Also update the local reference in all_detections for consistency within this function's current pass
                        larger_det_info['mask'] = detections_list[larger_det_info['category_idx']].mask[larger_det_info['detection_idx']]
                        
                        print(f"    Modified mask of {larger_det_info['category']} (cat_idx {larger_det_info['category_idx']}, det_idx {larger_det_info['detection_idx']}).")
                        continue # Conflict resolved for this pair, both kept (one modified)

                # --- Default/Fallback Priority Logic (if not handled by Cond 1 or Cond 2) ---
                if det1_info['priority'] > det2_info['priority']:
                    to_remove.add((det2_info['category_idx'], det2_info['detection_idx']))
                    print(f"  Action (Default): Removing Det2 ({det2_info['category']}, prio {det2_info['priority']}) - lower priority than Det1 ({det1_info['category']}, prio {det1_info['priority']}).")
                elif det2_info['priority'] > det1_info['priority']:
                    to_remove.add((det1_info['category_idx'], det1_info['detection_idx']))
                    print(f"  Action (Default): Removing Det1 ({det1_info['category']}, prio {det1_info['priority']}) - lower priority than Det2 ({det2_info['category']}, prio {det2_info['priority']}).")
                else: # Same priority
                    if area1 < area2:
                        to_remove.add((det1_info['category_idx'], det1_info['detection_idx']))
                        print(f"  Action (Default): Removing Det1 ({det1_info['category']}) - same priority, smaller area ({area1} vs {area2}).")
                    elif area2 < area1:
                        to_remove.add((det2_info['category_idx'], det2_info['detection_idx']))
                        print(f"  Action (Default): Removing Det2 ({det2_info['category']}) - same priority, smaller area ({area2} vs {area1}).")
                    else: # Same priority and same area
                        # Arbitrarily remove the second one in the comparison to ensure determinism
                        to_remove.add((det2_info['category_idx'], det2_info['detection_idx']))
                        print(f"  Action (Default): Removing Det2 ({det2_info['category']}) - same priority, same area. Arbitrary choice.")
    
    # Create filtered detection lists based on `to_remove`
    # The masks in `detections_list` might have been modified in-place.
    filtered_detections_list = []
    for i, (detections_obj, category_name) in enumerate(zip(detections_list, category_names)):
        if detections_obj.mask is None or len(detections_obj.mask) == 0:
            filtered_detections_list.append(detections_obj) # Keep if empty or no masks
            continue
        
        # Determine which indices to keep for the current category's detections
        indices_to_keep_for_this_category = []
        current_cat_removed_indices = {det_idx for cat_idx, det_idx in to_remove if cat_idx == i}

        for j in range(len(detections_obj.mask)):
            if j not in current_cat_removed_indices:
                indices_to_keep_for_this_category.append(j)
        
        if not indices_to_keep_for_this_category:
            filtered_detections_list.append(sv.Detections.empty())
        elif len(indices_to_keep_for_this_category) == len(detections_obj.mask): # No detections removed for this category
            filtered_detections_list.append(detections_obj)
        else:
            # Filter the sv.Detections object for this category
            # Create a boolean mask for filtering the Detections object
            keep_mask_for_sv_detections = np.zeros(len(detections_obj.mask), dtype=bool)
            keep_mask_for_sv_detections[indices_to_keep_for_this_category] = True
            filtered_detections_list.append(detections_obj[keep_mask_for_sv_detections])
            
    print(f"\nOverlap resolution complete. Initially {len(all_detections)} detections considered, {len(to_remove)} marked for removal.")
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
    confidence_threshold: float = 0.3 # This threshold is not used in current logic
) -> bool:
    """
    Validates a detection by cropping the region and using Florence2's region captioning
    to check if the detected object matches the expected category.
    """
    try:
        cropped_image = crop_image_from_bbox(image_pil, bbox)
        if cropped_image.width == 0 or cropped_image.height == 0:
            print(f"Warning: Cropped image for '{expected_category}' has zero dimension, rejecting detection.")
            return False
            
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=cropped_image,
            task=FLORENCE_DETAILED_CAPTION_TASK, # Using detailed caption for validation
            text="" # No text prompt needed for detailed caption
        )
        
        caption = result.get(FLORENCE_DETAILED_CAPTION_TASK, "").lower()
        
        if not caption:
            print(f"No caption generated for region of '{expected_category}', rejecting detection.")
            return False
        
        valid_categories = CATEGORY_VALIDATION_MAP.get(expected_category.lower(), [expected_category.lower()])
        
        matches = sum(1 for category_keyword in valid_categories if category_keyword.lower() in caption)
        
        is_valid = matches > 0
        
        # if is_valid:
        #     print(f"Validation for '{expected_category}': Caption='{caption}', Valid={is_valid}'")
        # else:
        #     print(f"Validation for '{expected_category}': Caption='{caption}', Valid={is_valid}")
        
        return is_valid
        
    except Exception as e:
        print(f"Error validating detection for '{expected_category}': {e}")
        return False


def filter_valid_detections(
    image_pil: Image.Image, 
    detections: sv.Detections, 
    expected_category: str
) -> sv.Detections:
    """
    Filters detections by validating each bounding box using region captioning.
    """
    if not detections.xyxy.any(): # More robust check for empty detections
        return detections
    
    valid_mask = np.zeros(len(detections.xyxy), dtype=bool)
    
    for i, bbox in enumerate(detections.xyxy):
        if validate_detection_with_region_caption(image_pil, bbox, expected_category):
            valid_mask[i] = True
        # else:
            # print(f"Rejected detection {i} for category '{expected_category}' after validation.") # Already printed in validate_detection
    
    if not np.any(valid_mask):
        return sv.Detections.empty()
    
    return detections[valid_mask]


@torch.inference_mode() #
def process_single_image(
    image_pil: Image.Image, text_prompt: str
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Processes a single image for open vocabulary detection and segmentation.
    Returns (annotated_image, mask_only_image)
    """
    use_cuda = DEVICE.type == "cuda"
    autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16} if use_cuda else {"device_type": "cpu", "dtype": torch.float32}

    with torch.autocast(**autocast_kwargs): #
        texts = [prompt.strip() for prompt in text_prompt.split(",")]
        initial_detections_list = []
        initial_category_names = [] 

        for text in texts:
            print(f"\nProcessing category: {text}")
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

            if detections.xyxy.any():
                print(f"Found {len(detections.xyxy)} initial detections for '{text}', validating...")
                detections = filter_valid_detections(image_pil, detections, text)
                print(f"After validation: {len(detections.xyxy)} valid detections for '{text}'")

                if detections.xyxy.any():
                    detections = run_sam_inference(SAM_IMAGE_MODEL, image_pil, detections)
                    if detections.mask is not None and len(detections.mask) > 0: 
                        initial_detections_list.append(detections)
                        initial_category_names.append(text) 
                    else:
                        print(f"No masks after SAM for '{text}', skipping this category's detections.")
                # else: # Already handled by print above
                #     print(f"No valid detections after validation for '{text}'")
            else:
                print(f"No initial detections found for '{text}'")

        if not initial_detections_list:
            print(f"No valid detections found for any prompt in the image after initial processing and SAM.")
            return None, None

        if len(initial_detections_list) > 0: # Only resolve if there's something to resolve
            print("\nStarting overlap resolution...")
            # Pass a copy of the list of detections if they are simple lists, 
            # but sv.Detections are objects, so modifications within resolve_overlapping_masks
            # to the mask arrays themselves will persist.
            resolved_detections_list = resolve_overlapping_masks(initial_detections_list, initial_category_names)
        else: # Should be caught by previous check, but good for safety
            resolved_detections_list = initial_detections_list


        final_detections_to_merge = []
        final_category_names_for_merge = []

        for i, dets in enumerate(resolved_detections_list):
            # initial_category_names[i] is the category for resolved_detections_list[i]
            # Ensure dets itself is not empty and has masks
            if dets.mask is not None and len(dets.mask) > 0 and dets.xyxy.any(): 
                final_detections_to_merge.append(dets)
                final_category_names_for_merge.append(initial_category_names[i]) 

        if not final_detections_to_merge:
            print(f"No valid detections remaining after overlap resolution and filtering.")
            return None, None

        merged_detections, all_labels = assign_consistent_colors_to_detections(
            final_detections_to_merge,
            final_category_names_for_merge 
        )

        if not (merged_detections.mask is not None and len(merged_detections.mask) > 0 and merged_detections.xyxy.any()):
            print(f"No objects with masks found after merging for prompt '{text_prompt}' in the image.")
            return None, None
        
        annotated_image = annotate_image_pil(image_pil, merged_detections, all_labels)
        mask_only_image = create_mask_only_image(image_pil, merged_detections)

        return annotated_image, mask_only_image

def main():
    start_time = time.time()
    create_output_directories()
    print(f"Reading image paths from: {TEST_FILE}")
    image_paths_data = read_test_file(TEST_FILE)
    
    if not image_paths_data:
        print(f"No valid images found in {TEST_FILE}. Please check the file and paths.")
        return

    print(f"Found {len(image_paths_data)} images to process.")

    for relative_path, full_path in image_paths_data:
        print(f"\n======================================================================")
        print(f"Processing image: {full_path}")
        print(f"======================================================================")
        try:
            image_pil = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {full_path}: {e}")
            continue

        annotated_image, mask_only_image = process_single_image(image_pil, FIXED_TEXT_PROMPT)

        if annotated_image and mask_only_image:
            try:
                annotated_output_path = get_output_path(relative_path, OUTPUT_ANNOTATED_FOLDER)
                annotated_image.save(annotated_output_path)
                print(f"Saved annotated image to: {annotated_output_path}")
                
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