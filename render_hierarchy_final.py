import math
import os
import torch
from random import randint
from utils.loss_utils import ssim, ssim_masked
from gaussian_renderer import render_post
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr, psnr_masked
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
from lpipsPyTorch import lpips
import torchvision.io # Added for reading segmentation masks
import numpy as np
# from PIL import ImageColor # Option for hex to rgb, or use a simpler custom function

from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights

CATEGORY_GROUPS = {
    "sky": {"color": "#87CEEB", "categories": ["sky"]},  # Sky Blue
    "ground": {"color": "#8B4513", "categories": ["ground", "floor", "road"]},  # Saddle Brown
    "buildings": {"color": "#696969", "categories": ["house", "building", "wall"]},  # Dim Gray
    "vehicles": {"color": "#FF4500", "categories": ["car", "bike"]},  # Orange Red
    "vegetation": {"color": "#228B22", "categories": ["vegetation", "plant"]},  # Forest Green
    "lamposts": {"color": "#FFD700", "categories": ["lampost"]}  # Gold
}

# EDIT 1: Added depth stratification ranges
DEPTH_RANGES = [
    ("near", 0.0, 5.0),      # 0-5 meters  
    ("medium", 5.0, 20.0),   # 5-20 meters
    ("far", 20.0, float('inf'))  # 20+ meters
]

# TEST FUNCTIONS

def save_depth_masks_debug(invdepth_map, alpha_mask, save_dir, image_name):
    """
    Save depth-stratified masks as PNG images for debugging.
    
    Args:
        invdepth_map: Inverse depth map from rendering
        alpha_mask: Valid pixel mask 
        save_dir: Directory to save debug images
        image_name: Name of the current image (for filename)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert inverse depth to regular depth for visualization
    epsilon = 1e-6
    depth_map = 1.0 / (invdepth_map + epsilon)
    
    # Create and save masks for each depth range
    for range_name, min_depth, max_depth in DEPTH_RANGES:
        # Get the depth range mask
        range_mask = get_depth_range_mask(invdepth_map, min_depth, max_depth)
        
        # Combine with alpha mask to show only valid pixels
        combined_mask = range_mask * alpha_mask
        
        # Convert to 0-255 range for PNG saving
        mask_vis = (combined_mask * 255).byte()
        
        # Save the mask
        mask_filename = f"{os.path.splitext(image_name)[0]}_depth_{range_name}_mask.png"
        mask_path = os.path.join(save_dir, mask_filename)
        if not os.path.exists(mask_path):
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        torchvision.utils.save_image(mask_vis.float() / 255.0, mask_path)
        
        print(f"Saved depth mask for {range_name} range: {mask_path}")
    
    # Also save a combined visualization showing all depth ranges in different colors
    save_combined_depth_visualization(invdepth_map, alpha_mask, save_dir, image_name)

def save_combined_depth_visualization(invdepth_map, alpha_mask, save_dir, image_name):
    """
    Save a combined visualization showing all depth ranges in different colors.
    """
    # Create RGB image to show different depth ranges
    h, w = invdepth_map.shape[-2:]
    combined_vis = torch.zeros(3, h, w, device=invdepth_map.device)
    
    # Define colors for each depth range (RGB values 0-1)
    colors = [
        [1.0, 0.0, 0.0],  # Red for near
        [0.0, 1.0, 0.0],  # Green for medium  
        [0.0, 0.0, 1.0],  # Blue for far
    ]
    
    for i, (range_name, min_depth, max_depth) in enumerate(DEPTH_RANGES):
        range_mask = get_depth_range_mask(invdepth_map, min_depth, max_depth)
        combined_mask = range_mask * alpha_mask
        
        # Apply color to this range
        for c in range(3):
            combined_vis[c] += combined_mask.squeeze() * colors[i][c]
    
    # Save combined visualization
    combined_filename = f"{os.path.splitext(image_name)[0]}_depth_ranges_combined.png"
    combined_path = os.path.join(save_dir, combined_filename)
    if not os.path.exists(combined_path):
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    torchvision.utils.save_image(combined_vis, combined_path)
    
    print(f"Saved combined depth visualization: {combined_path}")

# TEST FUNCTIONS

def hex_to_rgb_tensor(hex_color_str, device):
    """Converts a hex color string to an RGB tensor."""
    hex_color_str = hex_color_str.lstrip('#')
    r = int(hex_color_str[0:2], 16)
    g = int(hex_color_str[2:4], 16)
    b = int(hex_color_str[4:6], 16)
    return torch.tensor([r, g, b], dtype=torch.uint8, device=device)

# EDIT 3: Added helper function for depth stratification
def get_depth_range_mask(invdepth_map, min_depth, max_depth):
    """
    Create mask for pixels within a specific depth range.
    
    Args:
        invdepth_map: Inverse depth map
        alpha_mask: Valid pixel mask
        min_depth: Minimum depth (in actual distance units)
        max_depth: Maximum depth (in actual distance units)
    """
    # Convert inverse depth back to depth for range checking
    epsilon = 1e-6
    depth_map = 1.0 / (invdepth_map + epsilon)
    
    # Create range mask
    range_mask = (depth_map >= min_depth) & (depth_map < max_depth)
    range_mask = range_mask.float()
    
    return range_mask

# EDIT 4: Modified compute_metrics function
def compute_metrics(image, gt_image, alpha_mask, segmentation_mask=None):
    """Compute metrics with optional weighting."""
    masked_image = image * alpha_mask
    masked_gt_image = gt_image * alpha_mask
    
    if segmentation_mask is None:
        # Compute RGB metrics
        psnr_val = psnr(masked_image, masked_gt_image).mean().double()
        ssim_val = ssim(masked_image, masked_gt_image).mean().double()
        lpips_val = lpips(masked_image, masked_gt_image, net_type='vgg').mean().double()
    else:
        psnr_val = psnr_masked(masked_image, masked_gt_image, segmentation_mask).mean().double()
        ssim_val = ssim_masked(masked_image, masked_gt_image, segmentation_mask).mean().double()
        lpips_val = lpips(masked_image, masked_gt_image, net_type='vgg', mask=segmentation_mask).mean().double() # PACOMMENT: Implemented LPIPS with mask only for VGG
    
    return psnr_val, ssim_val, lpips_val

def compute_depth_metrics(pred_depth, gt_depth, mask):
    """Compute depth metrics with optional weighting."""
    num_valid_pixels = torch.sum(mask)
    
    if num_valid_pixels > 0:
        abs_diff = torch.abs(pred_depth - gt_depth) * mask
        squared_diff = torch.pow(pred_depth - gt_depth, 2) * mask
        
        imae = torch.sum(abs_diff) / num_valid_pixels
        irmse = torch.sqrt(torch.sum(squared_diff) / num_valid_pixels)
    else:
        imae = torch.tensor(0.0, device=pred_depth.device)
        irmse = torch.tensor(0.0, device=pred_depth.device)
    
    return imae.double(), irmse.double()

def direct_collate(x):
    return x

@torch.no_grad()
def render_set(args, scene, pipe, out_dir, tau, eval_mode): # Renamed eval to eval_mode to avoid conflict
    print(f"Rendering with tau: {tau}")
    render_path = out_dir

    render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    parent_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    nodes_for_render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
    interpolation_weights = torch.zeros(scene.gaussians._xyz.size(0)).float().cuda()
    num_siblings = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()

    # EDIT 5: Expanded metrics storage to include weighted and stratified metrics
    # Initialize metrics storage
    metrics = {
        "whole_image": {
            "unweighted": {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "imae": 0.0, "irmse": 0.0, "pixel_count": 0, "image_count": 0},
        }
    }
    
    # Add depth range metrics
    for range_name, _, _ in DEPTH_RANGES:
        metrics[f"depth_{range_name}"] = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "imae": 0.0, "irmse": 0.0, "pixel_count": 0, "image_count": 0}
    
    if args.segmentation_root_folder:
        for cat_name in CATEGORY_GROUPS.keys():
            metrics[cat_name] = {
                "unweighted": {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "imae": 0.0, "irmse": 0.0, "pixel_count": 0, "image_count": 0},
            }

    cameras = scene.getTestCameras() if eval_mode else scene.getTrainCameras()
    if not cameras:
        print(f"No cameras found for {'evaluation' if eval_mode else 'training'} set. Skipping render_set for tau {tau}.")
        return

    for viewpoint in tqdm(cameras, desc=f"Rendering Tau {tau}"):
        viewpoint=viewpoint
        viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
        viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
        viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
        viewpoint.camera_center = viewpoint.camera_center.cuda()

        tanfovx = math.tan(viewpoint.FoVx * 0.5)
        threshold = (2 * (tau + 0.5)) * tanfovx / (0.5 * viewpoint.image_width)

        to_render = expand_to_size(
            scene.gaussians.nodes,
            scene.gaussians.boxes,
            threshold,
            viewpoint.camera_center,
            torch.zeros((3)),
            render_indices,
            parent_indices,
            nodes_for_render_indices)
        
        indices = render_indices[:to_render].int().contiguous()
        node_indices = nodes_for_render_indices[:to_render].contiguous()

        get_interpolation_weights(
            node_indices,
            threshold,
            scene.gaussians.nodes,
            scene.gaussians.boxes,
            viewpoint.camera_center.cpu(), #This function might expect CPU tensor for camera_center
            torch.zeros((3)), #This function might expect CPU tensor for offset
            interpolation_weights, # Output, should be on device
            num_siblings # Output, should be on device
        )

        render_results = render_post(
            viewpoint, 
            scene.gaussians, 
            pipe, 
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"), 
            render_indices=indices,
            parent_indices=parent_indices,
            interpolation_weights=interpolation_weights,
            num_node_kids=num_siblings, 
            use_trained_exp=args.train_test_exp,
            do_depth=True
        )

        image = torch.clamp(render_results["render"], 0.0, 1.0)
        depth_image = render_results["depth"] 

        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        alpha_mask = viewpoint.alpha_mask.cuda() # Ensure alpha_mask is on the correct device and shape [1, H, W]

        if args.train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
            alpha_mask = alpha_mask[..., alpha_mask.shape[-1] // 2:]

        try:
            torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png"))
        except:
            os.makedirs(os.path.dirname(os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png")), exist_ok=True)
            torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png"))


        if eval_mode:
            gt_depth = viewpoint.invdepthmap.cuda() # Ensure gt_depth is on correct device
            
            # --- Whole Image Evaluation (Unweighted and Distance-Weighted) ---
            # Unweighted metrics
            psnr_unwt, ssim_unwt, lpips_unwt = compute_metrics(image, gt_image, alpha_mask, segmentation_mask=None)
            imae_unwt, irmse_unwt = compute_depth_metrics(depth_image, gt_depth, alpha_mask)

            num_pixels = torch.prod(torch.tensor(alpha_mask.shape)).item()
            
            metrics["whole_image"]["unweighted"]["psnr"] += psnr_unwt * num_pixels
            metrics["whole_image"]["unweighted"]["ssim"] += ssim_unwt * num_pixels
            metrics["whole_image"]["unweighted"]["lpips"] += lpips_unwt * num_pixels
            metrics["whole_image"]["unweighted"]["imae"] += imae_unwt * num_pixels
            metrics["whole_image"]["unweighted"]["irmse"] += irmse_unwt * num_pixels
            metrics["whole_image"]["unweighted"]["pixel_count"] += num_pixels
            metrics["whole_image"]["unweighted"]["image_count"] += 1

            # EDIT 7: Depth-Stratified Evaluation
            for range_name, min_depth, max_depth in DEPTH_RANGES:
                range_mask = get_depth_range_mask(gt_depth, min_depth, max_depth)
                
                if torch.sum(range_mask) > 0:  # Only evaluate if range has valid pixels
                    psnr_range, ssim_range, lpips_range = compute_metrics(image, gt_image, alpha_mask, segmentation_mask=range_mask)
                    imae_range, irmse_range = compute_depth_metrics(depth_image, gt_depth, alpha_mask*range_mask)
                    
                    valid_pixels = range_mask * alpha_mask
                    valid_pixels = valid_pixels > 0
                    num_pixels_range = torch.sum(valid_pixels).item()  # Count valid pixels in this range

                    metrics[f"depth_{range_name}"]["psnr"] += psnr_range * num_pixels_range
                    metrics[f"depth_{range_name}"]["ssim"] += ssim_range * num_pixels_range
                    metrics[f"depth_{range_name}"]["lpips"] += lpips_range * num_pixels_range
                    metrics[f"depth_{range_name}"]["imae"] += imae_range * num_pixels_range
                    metrics[f"depth_{range_name}"]["irmse"] += irmse_range * num_pixels_range
                    metrics[f"depth_{range_name}"]["pixel_count"] += num_pixels_range
                    metrics[f"depth_{range_name}"]["image_count"] += 1
            
            # DEBUGGING: Save depth-stratified masks
            if args.save_debug_masks:
                debug_dir = os.path.join(render_path, "debug_depth_masks")
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                save_depth_masks_debug(gt_depth, alpha_mask, debug_dir, viewpoint.image_name)

            # --- Per-Category Evaluation (if segmentation folder is provided) ---
            if args.segmentation_root_folder:
                
                # Construct path to segmentation mask
                # viewpoint.image_path is the full path to the original image
                # scene.dataset.source_path is the root for that scene (e.g., /path/to/dataset/scene1)
                
                segmentation_mask_name = os.path.splitext(viewpoint.image_name)[0] + ".png" # Assuming segmentation masks are named like this
                seg_mask_full_path = os.path.join(args.segmentation_root_folder, segmentation_mask_name)

                if not os.path.exists(seg_mask_full_path):
                    print(f"Warning: Segmentation mask not found for {viewpoint.image_name} at {seg_mask_full_path}. Skipping segmented evaluation for this image.")
                else:
                    seg_image_raw = torchvision.io.read_image(seg_mask_full_path).to("cuda") # Reads as [C, H, W], typically C=3 or 4 (RGBA)

                    if seg_image_raw.shape[0] == 4: # If RGBA, take RGB
                        seg_image_rgb = seg_image_raw[:3, :, :]
                    elif seg_image_raw.shape[0] == 3: # If RGB
                        seg_image_rgb = seg_image_raw
                    else:
                        print(f"Warning: Segmentation mask {seg_mask_full_path} has unexpected channel count: {seg_image_raw.shape[0]}. Expected 3 or 4. Skipping.")
                        continue # Skip to next viewpoint if seg mask format is wrong

                    # Ensure seg_image_rgb matches gt_image dimensions if necessary (it should by now)
                    if seg_image_rgb.shape[1:] != gt_image.shape[1:]:
                        print(f"Warning: Segmentation mask {seg_mask_full_path} ({seg_image_rgb.shape}) does not match image dimensions ({gt_image.shape}). Skipping segmented eval for this image.")
                        continue

                    for cat_name, cat_info in CATEGORY_GROUPS.items():
                        target_color_rgb = hex_to_rgb_tensor(cat_info["color"], "cuda") # Tensor [R, G, B] on device

                        # Create category mask (H, W)
                        category_mask_hw = (seg_image_rgb[0, :, :] == target_color_rgb[0]) & \
                                            (seg_image_rgb[1, :, :] == target_color_rgb[1]) & \
                                            (seg_image_rgb[2, :, :] == target_color_rgb[2])
                        
                        category_mask_1hw = category_mask_hw.unsqueeze(0).float() # Shape [1, H, W]

                        # Combine with alpha_mask for RGB metrics
                        num_pixels_in_cat_rgb = torch.sum(category_mask_1hw > 0)

                        if num_pixels_in_cat_rgb > 0:
                            # EDIT 8: Category evaluation with both unweighted and distance-weighted metrics
                            # Unweighted category metrics
                            psnr_cat_unwt, ssim_cat_unwt, lpips_cat_unwt = compute_metrics(
                                image, gt_image, alpha_mask, segmentation_mask=category_mask_1hw)
                        
                            # A pixel is valid is it is in the category mask and has a valid alpha value
                            valid_pixels = category_mask_1hw * alpha_mask
                            valid_pixels = valid_pixels > 0
                            pixel_count = torch.sum(valid_pixels).item()
                            
                            metrics[cat_name]["unweighted"]["psnr"] += psnr_cat_unwt * pixel_count
                            metrics[cat_name]["unweighted"]["ssim"] += ssim_cat_unwt * pixel_count
                            metrics[cat_name]["unweighted"]["lpips"] += lpips_cat_unwt * pixel_count
                            metrics[cat_name]["unweighted"]["pixel_count"] += pixel_count
                            metrics[cat_name]["unweighted"]["image_count"] += 1

                            # Unweighted depth metrics
                            imae_cat_unwt, irmse_cat_unwt = compute_depth_metrics(
                                depth_image, gt_depth, alpha_mask*category_mask_1hw)
                            
                            metrics[cat_name]["unweighted"]["imae"] += imae_cat_unwt * pixel_count
                            metrics[cat_name]["unweighted"]["irmse"] += irmse_cat_unwt * pixel_count


    torch.cuda.empty_cache()

    if eval_mode and len(cameras) > 0:
        # EDIT 9: Enhanced results reporting with all three evaluation types
        print(f"\n--- Results for TAU: {tau} ---")
        
        # Whole image metrics (unweighted vs distance-weighted)
        if metrics["whole_image"]["unweighted"]["image_count"] > 0:
            count = metrics["whole_image"]["unweighted"]["pixel_count"]
            count_images = metrics["whole_image"]["unweighted"]["image_count"]
            psnr_val = metrics["whole_image"]["unweighted"]["psnr"] / count
            ssim_val = metrics["whole_image"]["unweighted"]["ssim"] / count
            lpips_val = metrics["whole_image"]["unweighted"]["lpips"] / count
            imae_val = metrics["whole_image"]["unweighted"]["imae"] / count
            irmse_val = metrics["whole_image"]["unweighted"]["irmse"] / count
            print(f"Whole Image: PSNR: {psnr_val:.5f} SSIM: {ssim_val:.5f} LPIPS: {lpips_val:.5f} iMAE: {imae_val:.5f} iRMSE: {irmse_val:.5f} (pixel-weighted and evaluated on {count_images} images)")

        # Depth-stratified metrics
        print("\nDepth-Stratified Results:")
        for range_name, min_depth, max_depth in DEPTH_RANGES:
            metric_key = f"depth_{range_name}"
            if metrics[metric_key]["image_count"] > 0:
                count = metrics[metric_key]["pixel_count"]
                count_images = metrics[metric_key]["image_count"]
                psnr_val = metrics[metric_key]["psnr"] / count
                ssim_val = metrics[metric_key]["ssim"] / count
                lpips_val = metrics[metric_key]["lpips"] / count
                imae_val = metrics[metric_key]["imae"] / count
                irmse_val = metrics[metric_key]["irmse"] / count
                depth_range_str = f"{min_depth}-{max_depth}m" if max_depth != float('inf') else f"{min_depth}m+"
                print(f"  {range_name.capitalize()} ({depth_range_str}): PSNR: {psnr_val:.5f} SSIM: {ssim_val:.5f} LPIPS: {lpips_val:.5f} iMAE: {imae_val:.5f} iRMSE: {irmse_val:.5f} (pixel-weighted and evaluated on {count_images} instances)")
            else:
                depth_range_str = f"{min_depth}-{max_depth}m" if max_depth != float('inf') else f"{min_depth}m+"
                print(f"  {range_name.capitalize()} ({depth_range_str}): No valid pixels found in this depth range.")

        # Per-category metrics (unweighted vs distance-weighted)
        if args.segmentation_root_folder:
            print("\nCategory-Specific Results:")
            for cat_name in CATEGORY_GROUPS.keys():
                if metrics[cat_name]["unweighted"]["image_count"] > 0:
                    count = metrics[cat_name]["unweighted"]["pixel_count"]
                    count_images = metrics[cat_name]["unweighted"]["image_count"]
                    psnr_val = metrics[cat_name]["unweighted"]["psnr"] / count
                    ssim_val = metrics[cat_name]["unweighted"]["ssim"] / count
                    lpips_val = metrics[cat_name]["unweighted"]["lpips"] / count
                    imae_val = metrics[cat_name]["unweighted"]["imae"] / count
                    irmse_val = metrics[cat_name]["unweighted"]["irmse"] / count
                    print(f"  Category '{cat_name}' : PSNR: {psnr_val:.5f} SSIM: {ssim_val:.5f} LPIPS: {lpips_val:.5f} iMAE: {imae_val:.5f} iRMSE: {irmse_val:.5f} (pixel-weighted and evaluated on {count_images} images)")
                else:
                    print(f"  Category '{cat_name}' : Not found or no valid overlap in any evaluated image.")
        print("-----------------------------------\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser) # OptimizationParams not used in this script directly but good practice if it were extended
    pp = PipelineParams(parser)
    parser.add_argument('--out_dir', type=str, default="renders_output", help="Directory to save rendered images and results.")
    parser.add_argument("--taus", nargs="+", type=float, default=[0.0])
    # Added argument for segmentation folder
    parser.add_argument('--segmentation_root_folder', type=str, default=None, 
                        help="Root folder for segmentation masks. Expected to mirror the structure of the image dataset relative to its source_path.")

    parser.add_argument('--save_debug_masks', action='store_true', 
                        help="Save depth-stratified masks as PNG images for debugging.")
    # PACOMMENT: Removed the commented out --images, assuming ModelParams handles image paths.
    args = parser.parse_args(sys.argv[1:])
    
    print("Rendering " + args.model_path)
    if args.segmentation_root_folder:
        print("Segmentation root folder: " + args.segmentation_root_folder)
        os.makedirs(args.segmentation_root_folder, exist_ok=True)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    dataset_params, pipe_params = lp.extract(args), pp.extract(args) # Corrected to use lp and pp
    
    gaussians = GaussianModel(dataset_params.sh_degree)
    gaussians.active_sh_degree = dataset_params.sh_degree # Use dataset_params
    
    # Pass dataset_params to Scene constructor
    scene = Scene(dataset_params, gaussians, resolution_scales=[1.0], create_from_hier=True) # Use 1.0 for full res

    for tau_val in args.taus:
        render_set(args, scene, pipe_params, os.path.join(args.out_dir, f"render_tau_{tau_val}"), tau_val, args.eval)