'''
Thesis Project: Street-sparse-3DGS
Author: Iacopo Ermacora
Date: 11/2024-06/2025

Description: This script converts a PLY mesh into CTM meshes split into grid cells 
(as required by the Cyclomedia pipeline) using the OpenCTM C++ library.
'''
import argparse
import os
import numpy as np
import trimesh
from collections import defaultdict
import ctm_exporter  # Import the C++ module built
from tqdm import tqdm
import json

def save_ctm(filename, vertices, triangles):
    """
    Save the mesh to a CTM file using the ctm_exporter C++ module.
    
    Args:
      - filename: output file path
      - vertices: (n,3) numpy array of float32 vertices
      - triangles: (m,3) numpy array of uint32 triangle indices
    """
    # Ensure vertices are float32 and triangles are uint32.
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    triangles = np.ascontiguousarray(triangles, dtype=np.uint32)
    ctm_exporter.save_ctm(filename, vertices, triangles)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert a PLY mesh into CTM meshes split into grid cells using OpenCTM C++ library."
    )
    parser.add_argument("--project_dir", type=str, required=True,
                        help="Path to the project directory.")
    parser.add_argument("--input_ply", required=True,
                        help="Input PLY mesh file.")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory where CTM files will be written.")
    parser.add_argument("--cell_size", type=float, default=50.0,
                        help="Size (in mesh coordinate units) of a grid cell. (Default: 50.0)")
    parser.add_argument("--cell_prefix", type=str, default="pmg__hires_",
                        help="Prefix for the output CTM file names. (Default: 'pmg__hires_')")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the mesh using trimesh.
    mesh = trimesh.load(args.input_ply, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        # Merge multiple meshes if necessary.
        mesh = trimesh.util.concatenate(mesh.dump())
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    
    # Apply translation to the mesh vertices based on the translation.json file
    with open(f"{args.project_dir}/camera_calibration/translation.json", "r") as f:
        translation_values = json.load(f)
    translation = np.array([translation_values['x_translation'], translation_values['y_translation'], 0], dtype=np.float32)
    vertices += translation
    
    # Group faces by grid cell (based on x and y coordinates of face centroids).
    cell_faces = defaultdict(list)
    for face in tqdm(faces):
        face_vertices = vertices[face]
        centroid = face_vertices.mean(axis=0)
        cell_x = int(np.floor(centroid[0] / args.cell_size))
        cell_y = int(np.floor(centroid[1] / args.cell_size))
        cell_faces[(cell_x, cell_y)].append(face)
    
    # For each cell, reindex vertices and export using the ctm_exporter module.
    for (cx, cy), face_list in tqdm(cell_faces.items()):
        cell_face_array = np.array(face_list, dtype=np.int32)
        unique_indices, inverse_indices = np.unique(cell_face_array, return_inverse=True)
        cell_vertices = vertices[unique_indices]
        cell_triangles = inverse_indices.reshape((-1, 3))
        
        out_filename = os.path.join(args.output_dir,
                                    f"{args.cell_prefix}{cx}_{cy}.ctm")
        print(f"Writing cell ({cx}, {cy}) with {cell_vertices.shape[0]} vertices and "
              f"{cell_triangles.shape[0]} faces to {out_filename}")
        save_ctm(out_filename, cell_vertices, cell_triangles)
