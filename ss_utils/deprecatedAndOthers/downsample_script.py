import laspy
import numpy as np
from tqdm import tqdm
import os

# --- Parameters ---
current_dir = os.path.dirname(os.path.abspath(__file__))
input_file  = f"{current_dir}/original.laz"
output_file = f"{current_dir}/downsampled.laz"

# --- Read the LAZ file using laspy 2.0 ---
las = laspy.read(input_file)

print(f"Read {len(las.points)} points from {input_file}")

# --- Get coordinates and compute spatial extents ---
X = las.x
Y = las.y
Z = las.z

min_x, max_x = X.min(), X.max()
min_y, max_y = Y.min(), Y.max()
min_z, max_z = Z.min(), Z.max()

# Define cube side length as (x extent)/100.
cell_size = (max_x - min_x) / 50.0

print(f"Cell (cube) side length: {cell_size}")

# --- Assign points to 3D cubes ---
# Use the same cell size for X, Y, and Z.
cell_x = ((X - min_x) / cell_size).astype(int)
cell_y = ((Y - min_y) / cell_size).astype(int)
cell_z = ((Z - min_z) / cell_size).astype(int)

# Stack cell indices to create a (N, 3) array.
cells = np.column_stack((cell_x, cell_y, cell_z))

# --- Compute density per cube ---
# Get unique cubes along with an inverse mapping and counts.
unique_cells, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
max_count = counts.max()
target_density = max_count / 100

print(f"Maximum points in a cube: {max_count}")
print(f"Target density (max_count / 10): {target_density}")

# --- Downsample points per cube ---
# Create a boolean mask for points to keep.
keep = np.full(len(X), False)

# Loop over each unique cube with a tqdm progress bar.
for cell_idx, cell_count in tqdm(enumerate(counts), total=len(counts), desc="Processing cubes"):
    pts_in_cube = np.where(inverse == cell_idx)[0]
    if cell_count > target_density:
        # Randomly select int(target_density) points.
        sampled = np.random.choice(pts_in_cube, int(target_density), replace=False)
        keep[sampled] = True
    else:
        keep[pts_in_cube] = True

print(f"Keeping {np.count_nonzero(keep)} out of {len(X)} points.")

# --- Create and write the downsampled LAZ file ---
# Create a new LAS file using the header information from the original file.
# Note: laspy 2.0 uses laspy.create() with a point_format and file_version.
new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
# Copy the header from the original file.
new_las.header = las.header
# Copy the points that were kept.
new_las.points = las.points[keep]

new_las.write(output_file)
print(f"Filtered point cloud written to {output_file}")

