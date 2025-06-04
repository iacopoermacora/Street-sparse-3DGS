import math
import os
import torch
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render_post
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr
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

def hex_to_rgb_tensor(hex_color_str, device):
    """Converts a hex color string to an RGB tensor."""
    hex_color_str = hex_color_str.lstrip('#')
    r = int(hex_color_str[0:2], 16)
    g = int(hex_color_str[2:4], 16)
    b = int(hex_color_str[4:6], 16)
    return torch.tensor([r, g, b], dtype=torch.uint8, device=device)

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

    # Initialize metrics storage
    metrics = {
        "whole_image": {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "imae": 0.0, "irmse": 0.0, "count": 0}
    }
    if args.segmentation_root_folder:
        for cat_name in CATEGORY_GROUPS.keys():
            metrics[cat_name] = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "imae": 0.0, "irmse": 0.0, "count": 0}

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
            # --- Whole Image Evaluation ---
            # Ensure alpha_mask has compatible dimensions for broadcasting (e.g., [1, H, W] or [H,W])
            # image and gt_image are [C, H, W]. alpha_mask is usually [1, H, W] or [H,W]
            # If alpha_mask is [H,W], unsqueeze it: alpha_mask = alpha_mask.unsqueeze(0) if alpha_mask.dim() == 2 else alpha_mask
            # Assuming alpha_mask is already [1, H, W] from viewpoint loader.
            
            masked_image_whole = image * alpha_mask
            masked_gt_image_whole = gt_image * alpha_mask
            
            metrics["whole_image"]["psnr"] += psnr(masked_image_whole, masked_gt_image_whole).mean().double()
            metrics["whole_image"]["ssim"] += ssim(masked_image_whole, masked_gt_image_whole).mean().double()
            metrics["whole_image"]["lpips"] += lpips(masked_image_whole, masked_gt_image_whole, net_type='vgg').mean().double()
            metrics["whole_image"]["count"] += 1
            
            gt_depth = viewpoint.invdepthmap.cuda() # Ensure gt_depth is on correct device
            valid_mask = (alpha_mask > 0).float() # This will be [1, H, W]
            
            masked_pred_depth_whole = depth_image * valid_mask
            masked_gt_depth_whole = gt_depth * valid_mask
            
            num_valid_pixels_whole = torch.sum(valid_mask)
            if num_valid_pixels_whole > 0:
                abs_diff_whole = torch.abs(masked_pred_depth_whole - masked_gt_depth_whole)
                imae_whole = torch.sum(abs_diff_whole) / num_valid_pixels_whole
                
                squared_diff_whole = torch.pow(masked_pred_depth_whole - masked_gt_depth_whole, 2)
                irmse_whole = torch.sqrt(torch.sum(squared_diff_whole) / num_valid_pixels_whole)
                
                metrics["whole_image"]["imae"] += imae_whole.double()
                metrics["whole_image"]["irmse"] += irmse_whole.double()
            else: # Should not happen if alpha_mask has some True values
                metrics["whole_image"]["imae"] += 0.0 
                metrics["whole_image"]["irmse"] += 0.0


            # --- Per-Category Evaluation (if segmentation folder is provided) ---
            if args.segmentation_root_folder:
                try:
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
                            final_rgb_eval_mask = alpha_mask * category_mask_1hw # Element-wise product
                            num_pixels_in_cat_rgb = torch.sum(final_rgb_eval_mask > 0)

                            if num_pixels_in_cat_rgb > 0:
                                metrics[cat_name]["count"] += 1 # Count this category instance

                                # RGB metrics for category
                                masked_image_cat = image * final_rgb_eval_mask
                                masked_gt_image_cat = gt_image * final_rgb_eval_mask
                                
                                metrics[cat_name]["psnr"] += psnr(masked_image_cat, masked_gt_image_cat).mean().double()
                                metrics[cat_name]["ssim"] += ssim(masked_image_cat, masked_gt_image_cat).mean().double()
                                metrics[cat_name]["lpips"] += lpips(masked_image_cat, masked_gt_image_cat, net_type='vgg').mean().double()

                                # Depth metrics for category
                                # valid_mask is (alpha_mask > 0).float(), shape [1, H, W]
                                final_depth_eval_mask = valid_mask * category_mask_1hw # [1, H, W]
                                num_pixels_in_cat_depth = torch.sum(final_depth_eval_mask > 0)

                                if num_pixels_in_cat_depth > 0:
                                    # depth_image and gt_depth are [1, H, W]
                                    abs_diff_cat = torch.abs(depth_image - gt_depth) * final_depth_eval_mask # Element-wise product, then sum over valid region
                                    imae_cat = torch.sum(abs_diff_cat) / num_pixels_in_cat_depth
                                    
                                    squared_diff_cat = torch.pow(depth_image - gt_depth, 2) * final_depth_eval_mask
                                    irmse_cat = torch.sqrt(torch.sum(squared_diff_cat) / num_pixels_in_cat_depth)
                                    
                                    metrics[cat_name]["imae"] += imae_cat.double()
                                    metrics[cat_name]["irmse"] += irmse_cat.double()
                                else: # No valid depth pixels for this category in this image
                                    metrics[cat_name]["imae"] += 0.0 
                                    metrics[cat_name]["irmse"] += 0.0
                            # else: category not present or not overlapping with alpha_mask in this image for RGB, its count won't be incremented.
                
                except Exception as e:
                    print(f"Error processing segmentation for {viewpoint.image_name}: {e}. Skipping segmented evaluation for this image.")


    torch.cuda.empty_cache()

    if eval_mode and len(cameras) > 0:
        print(f"\n--- Results for TAU: {tau} ---")
        # Average and print whole image metrics
        if metrics["whole_image"]["count"] > 0:
            count = metrics["whole_image"]["count"]
            psnr_val = metrics["whole_image"]["psnr"] / count
            ssim_val = metrics["whole_image"]["ssim"] / count
            lpips_val = metrics["whole_image"]["lpips"] / count
            imae_val = metrics["whole_image"]["imae"] / count # Already normalized per image, then averaged
            irmse_val = metrics["whole_image"]["irmse"] / count # Already normalized per image, then averaged
            print(f"Whole Image: PSNR: {psnr_val:.5f} SSIM: {ssim_val:.5f} LPIPS: {lpips_val:.5f} iMAE: {imae_val:.5f} iRMSE: {irmse_val:.5f} (evaluated on {count} images)")

        # Average and print per-category metrics
        if args.segmentation_root_folder:
            for cat_name in CATEGORY_GROUPS.keys():
                if metrics[cat_name]["count"] > 0:
                    count = metrics[cat_name]["count"]
                    psnr_val = metrics[cat_name]["psnr"] / count
                    ssim_val = metrics[cat_name]["ssim"] / count
                    lpips_val = metrics[cat_name]["lpips"] / count
                    imae_val = metrics[cat_name]["imae"] / count
                    irmse_val = metrics[cat_name]["irmse"] / count
                    print(f"Category '{cat_name}': PSNR: {psnr_val:.5f} SSIM: {ssim_val:.5f} LPIPS: {lpips_val:.5f} iMAE: {imae_val:.5f} iRMSE: {irmse_val:.5f} (evaluated on {count} instances)")
                else:
                    print(f"Category '{cat_name}': Not found or no valid overlap in any evaluated image.")
        print("-----------------------------------\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser) # OptimizationParams not used in this script directly but good practice if it were extended
    pp = PipelineParams(parser)
    parser.add_argument('--out_dir', type=str, default="renders_output", help="Directory to save rendered images and results.")
    parser.add_argument("--taus", nargs="+", type=float, default=[0.0, 3.0, 6.0, 15.0])
    # Added argument for segmentation folder
    parser.add_argument('--segmentation_root_folder', type=str, default=None, 
                        help="Root folder for segmentation masks. Expected to mirror the structure of the image dataset relative to its source_path.")
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