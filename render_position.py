# render_position.py

import os
import math
import torch
import numpy as np
import torchvision
from tqdm import tqdm
from gaussian_renderer import render_post
from utils.image_utils import psnr
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser
import sys

@torch.no_grad()
def render_one_group_shifted(args, scene, pipe, out_dir, new_x, new_y, new_z):
    cameras = scene.getTestCameras()
    print(f"Found {len(cameras)} test cameras")

    # --- Find first group of cameras with same position (T) ---
    grouped = {}
    z_positions = []
    for cam in cameras:
        camera_center = cam.camera_center.cpu().numpy()
        z_positions.append(camera_center[2])
        key = tuple(np.round(camera_center, decimals=2))
        grouped.setdefault(key, []).append(cam)
    
    average_z = np.mean(z_positions)

    if new_x is not None and new_y is not None:
        # Order the groups by the key value
        grouped = {k: v for k, v in sorted(grouped.items(), key=lambda item: item[0])}

    # Take the first group with 10 cameras
    grouped = {k: v for k, v in grouped.items() if len(v) >= 1}

    # Print camera name and coordinates for each group
    for position, cameras_in_group in grouped.items():
        # Get the first camera in this group to display its name
        representative_camera = cameras_in_group[0]
        print(f"Position {position}: {len(cameras_in_group)} cameras")
        print(f"  Sample camera name: {representative_camera.image_name}")
        print(f"  X: {position[0]:.4f}, Y: {position[1]:.4f}, Z: {position[2]:.4f}")
        print("------")
    
    chosen_key = next(iter(grouped))  # take the first group
    group = grouped[chosen_key]
    print(f"Using camera group with center = {chosen_key}, {len(group)} views")

    # --- Create output dir ---
    os.makedirs(out_dir, exist_ok=True)

    first_cam = True

    for cam in tqdm(group):
        if first_cam:
            print(f"Image name: {cam.image_name}")
            first_cam = False
        
        if new_x is None or new_y is None:
            new_x_trans = 0.0
            new_y_trans = 0.0
            new_z_trans = 0.0
        else:
            new_x_trans = new_x + (- cam.camera_center[0].item())
            new_y_trans = new_y + (- cam.camera_center[1].item())
            if new_z is None:
                new_z_trans = average_z + (- cam.camera_center[2].item())
            else:
                new_z_trans = new_z + (- cam.camera_center[2].item())
        # Modify the camera position only on x and y
        cam.trans = np.array([new_x_trans, new_y_trans, new_z_trans])
        cam.world_view_transform = torch.tensor(
            getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)
        ).transpose(0, 1).to(cam.data_device)

        cam.full_proj_transform = (
            cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
        ).squeeze(0).to(cam.data_device)

        cam.camera_center = cam.world_view_transform.inverse()[3, :3].to(cam.data_device)

        cam = cam
        cam.world_view_transform = cam.world_view_transform.cuda()
        cam.projection_matrix = cam.projection_matrix.cuda()
        cam.full_proj_transform = cam.full_proj_transform.cuda()
        cam.camera_center = cam.camera_center.cuda()

        # Prepare buffers
        render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
        parent_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
        nodes_for_render_indices = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()
        interpolation_weights = torch.zeros(scene.gaussians._xyz.size(0)).float().cuda()
        num_siblings = torch.zeros(scene.gaussians._xyz.size(0)).int().cuda()

        tau = 0.0
        tanfovx = math.tan(cam.FoVx * 0.5)
        threshold = (2 * (tau + 0.5)) * tanfovx / (0.5 * cam.image_width)

        to_render = expand_to_size(
            scene.gaussians.nodes, 
            scene.gaussians.boxes, 
            threshold,
            cam.camera_center, 
            torch.zeros((3)), 
            render_indices, 
            parent_indices,
            nodes_for_render_indices
        )

        indices = render_indices[:to_render].int().contiguous()
        node_indices = nodes_for_render_indices[:to_render].contiguous()

        get_interpolation_weights(
            node_indices, 
            threshold, 
            scene.gaussians.nodes, 
            scene.gaussians.boxes,
            cam.camera_center.cpu(), 
            torch.zeros((3)),
            interpolation_weights, 
            num_siblings
        )

        render_result = render_post(
            cam, 
            scene.gaussians, 
            pipe,
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"),
            render_indices=indices, 
            parent_indices=parent_indices,
            interpolation_weights=interpolation_weights,
            num_node_kids=num_siblings,
            use_trained_exp=False,
            do_depth=True  # Enable depth rendering
        )

        # Get the rendered image and depth
        image = torch.clamp(render_result["render"], 0.0, 1.0)
        depth = render_result["depth"]  # Get the depth map

        # Save the rendered image
        base_name = str(new_x) + "_" + str(new_y) + "_" + cam.image_name.split(".")[0].split("_")[-1]
        rgb_path = os.path.join(out_dir, base_name + ".png")
        torchvision.utils.save_image(image, rgb_path)
        
        # Save the depth map
        # Scale depth for better visualization (optional)
        # You can adjust this based on your scene's depth range
        os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
        depth_vis = depth / depth.max()  # Normalize to [0,1]
        depth_path = os.path.join(out_dir, "depth", base_name + "_depth.png")
        torchvision.utils.save_image(depth_vis.repeat(3, 1, 1), depth_path)  # Save as grayscale image

if __name__ == "__main__":
    parser = ArgumentParser()
    mp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--new_x", type=float)
    parser.add_argument("--new_y", type=float)
    parser.add_argument("--new_z", type=float)
    args = parser.parse_args(sys.argv[1:])

    if args.new_x is None or args.new_y is None:
        print("Value x and y not provided, displaying a random test image")
    dataset, pipe = mp.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    scene = Scene(dataset, gaussians, resolution_scales=[1], create_from_hier=True)

    render_one_group_shifted(args, scene, pipe, args.out_dir, args.new_x, args.new_y, args.new_z)
