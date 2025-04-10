'''
Thesis Project: Street-sparse-3DGS
Author: Iacopo Ermacora
Date: 11/2024-06/2025

Description: This script converts COLMAP binary model files (cameras.bin, images.bin) to
a custom JSON format required by vis2mesh.
'''
import os
import json
import numpy as np
import argparse
from read_write_model import read_cameras_binary, read_images_binary, qvec2rotmat

def add_item(image, cameras, output):
    """
    Processes a single image and adds its data to the output dictionary.
    
    Args:
        image: The image object from COLMAP.
        cameras: The dictionary of cameras from COLMAP.
        output: The output dictionary to which the image data will be added.
    """
    # Use COLMAP's qvec2rotmat to obtain the 3x3 rotation matrix.
    R = qvec2rotmat(image.qvec)
    # Ensure tvec is a column vector.
    t = np.array(image.tvec).reshape(3, 1)
    # Compute the camera center: C = -R^T * t.
    C = (-np.dot(R.T, t)).flatten().tolist()
    
    # Retrieve the camera corresponding to this image.
    camera = cameras[image.camera_id]
    # For a PINHOLE camera, COLMAP stores parameters as [f, cx, cy].
    f = camera.params[0]
    cx = camera.params[1]
    cy = camera.params[2]
    K = [
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ]
    
    entry = {
        "C": C,
        "K": K,
        "R": R.tolist(),
        "width": camera.width,
        "height": camera.height
    }
    output["imgs"].append(entry)

def convert_colmap_bin_to_json(model_dir, output_file):
    """
    Reads COLMAP binary model files from model_dir (expecting cameras.bin and images.bin)
    and converts them into a custom JSON format using COLMAP's own conversion functions.
    
    For each image, the script computes:
      - "R": the rotation matrix using qvec2rotmat from COLMAP.
      - "C": the camera center computed as: C = - R^T * t.
      - "K": the intrinsic matrix (here assuming a PINHOLE model with parameters [f, cx, cy]).
      - "width" and "height": from the camera model.
    """
    cameras_path = os.path.join(model_dir, "cameras.bin")
    # Add both images from images.bin and images_depths.bin
    images_path = os.path.join(model_dir, "images.bin")
    images_depth_path = os.path.join(model_dir, "images_depths.bin")
    
    # Read the binary model files using COLMAP's provided functions.
    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)
    images_depth = read_images_binary(images_depth_path)
    
    output = {"imgs": []}

    # List of accepted faces
    accepted_faces = ["f1", "l1", "r1", "b1", "u1", "u2"]
    
    # Process images
    for image_id, image in images.items():
        # Check if the image has a valid face
        # Extract the face name from the image name
        face_name = image.name.split("_")[-1].split(".")[0]
        if face_name in accepted_faces:
            add_item(image, cameras, output)
    
    # Process images_depth
    for image_id, image in images_depth.items():
        # Check if the image has a valid face
        # Extract the face name from the image name
        face_name = image.name.split("_")[-1].split(".")[0]
        if face_name in accepted_faces:
            add_item(image, cameras, output)

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Conversion complete. Output written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert COLMAP binary model to a custom JSON format using COLMAP functions for conversions."
    )
    parser.add_argument("--model_dir", help="Directory containing COLMAP binary model files (cameras.bin, images.bin)")
    parser.add_argument("--output_file", help="Path to output JSON file")
    args = parser.parse_args()
    
    convert_colmap_bin_to_json(args.model_dir, args.output_file)

