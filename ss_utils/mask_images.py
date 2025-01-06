import os
import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import struct

# Utility functions for reading COLMAP binary files
def read_cameras_binary(file_path):
    cameras = {}
    with open(file_path, "rb") as f:
        num_cameras = struct.unpack("<I", f.read(4))[0]
        for _ in range(num_cameras):
            cam_id, model, width, height = struct.unpack("<iiQQ", f.read(24))
            params = struct.unpack("<" + "d" * 4, f.read(32))
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras

def read_images_binary(file_path):
    images = {}
    with open(file_path, "rb") as f:
        num_images = struct.unpack("<I", f.read(4))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            tx, ty, tz = struct.unpack("<ddd", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                char = f.read(1)
                if not char or char == b"\0":
                    break
                name += char.decode("utf-8")
            
            if f.readable() and f.tell() + 4 <= os.fstat(f.fileno()).st_size:
                num_points = struct.unpack("<I", f.read(4))[0]
            else:
                num_points = 0  # Default to no points if the structure is incomplete
            
            xys = []
            point3D_ids = []
            for _ in range(num_points):
                if f.readable() and f.tell() + 24 <= os.fstat(f.fileno()).st_size:
                    x, y, point3D_id = struct.unpack("<ddq", f.read(24))
                    xys.append((x, y))
                    point3D_ids.append(point3D_id)
                else:
                    break  # Stop if the file doesn't contain enough bytes
            
            images[image_id] = {
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "camera_id": camera_id,
                "name": name,
                "xys": np.array(xys),
                "point3D_ids": np.array(point3D_ids),
            }
    return images


def read_points3D_binary(file_path):
    points3D = {}
    with open(file_path, "rb") as f:
        num_points = struct.unpack("<I", f.read(4))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("<q", f.read(8))[0]
            x, y, z = struct.unpack("<ddd", f.read(24))
            r, g, b = struct.unpack("<BBB", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<I", f.read(4))[0]
            points3D[point3D_id] = {
                "x": x, "y": y, "z": z,
                "color": (r, g, b),
                "error": error,
            }
    return points3D

def compute_sfm_point_density(mask, xys, point3D_ids, points3D):
    mask_coords = np.argwhere(mask > 0)
    valid_points = []
    for coord in mask_coords:
        y, x = coord
        distances = np.linalg.norm(xys - [x, y], axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < 1.5:
            point3D_id = point3D_ids[closest_idx]
            if point3D_id != -1 and point3D_id in points3D:
                valid_points.append(points3D[point3D_id]["error"])
    valid_points = [e for e in valid_points if e <= 1.5]
    return len(valid_points) / len(mask_coords) if mask_coords.size > 0 else 0

# Load the Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

def generate_masks(image_folder, output_folder, cameras_file, images_file, points3D_file):
    os.makedirs(output_folder, exist_ok=True)

    cameras = read_cameras_binary(cameras_file)
    images = read_images_binary(images_file)
    points3D = read_points3D_binary(points3D_file)

    for cam_folder in os.listdir(image_folder):
        input_path = os.path.join(image_folder, cam_folder)
        output_path = os.path.join(output_folder, cam_folder)
        os.makedirs(output_path, exist_ok=True)

        for image_name in os.listdir(input_path):
            image_path = os.path.join(input_path, image_name)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

            # Get predictions from Mask R-CNN
            with torch.no_grad():
                predictions = model(image_tensor)[0]

            masks = predictions["masks"] > 0.5
            labels = predictions["labels"]

            # Select relevant categories (person, animal, vehicle)
            relevant_labels = {1, 17, 18, 19, 20, 21, 22}  # COCO classes
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

            img_info = next((img for img in images.values() if img["name"] == os.path.join(cam_folder, image_name)), None)
            if img_info:
                for mask, label in zip(masks, labels):
                    if label.item() in relevant_labels:
                        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8) * 255

                        # Check SfM density for this mask
                        sfm_density = compute_sfm_point_density(
                            mask_np, img_info["xys"], img_info["point3D_ids"], points3D
                        )

                        # Remove static vehicles based on density
                        if label.item() in {3, 6, 8} and sfm_density >= 0.1:  # Vehicles
                            continue

                        combined_mask[mask_np > 0] = 255

            # Save mask
            mask_path = os.path.join(output_path, image_name)
            cv2.imwrite(mask_path, combined_mask)
            break

# Example usage
generate_masks(
    image_folder="/host/inputs/images",
    output_folder="/host/inputs/masks",
    cameras_file="/host/camera_calibration/unrectified/sparse/0/cameras.bin",
    images_file="/host/camera_calibration/unrectified/sparse/0/images.bin",
    points3D_file="/host/camera_calibration/unrectified/sparse/0/points3D.bin"
)
