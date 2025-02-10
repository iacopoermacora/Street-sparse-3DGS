import numpy as np
import laspy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os
import datetime

def load_laz(file_path):
    """Load a .laz point cloud file."""
    with laspy.open(file_path) as las:
        points = las.read()
    print(f"Loaded {len(points)} points from {file_path}")

    # downsampled_points = points[::100]

    # points = downsampled_points

    xyz = np.vstack([points.x, points.y, points.z]).T
    rgb = np.vstack([points.red, points.green, points.blue]).T
    print("Loaded xyz and rgb")
    return xyz, rgb, points

def cluster_points(xyz, rgb, eps=0.2, min_samples=50):
    """Perform DBSCAN clustering based on spatial and colour features."""
    features = np.hstack([xyz, rgb / 255.0])  # Normalise RGB values
    features = StandardScaler().fit_transform(features)
    print("Fitting DBSCAN")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    print("DBSCAN complete")
    return clustering.labels_

def recolour_clusters(labels, rgb):
    """Assign random colours to clusters for visualisation."""
    print("Recolouring clusters")
    unique_labels = np.unique(labels)
    cluster_colours = {label: np.random.randint(0, 255, 3) for label in unique_labels if label != -1}
    recoloured = np.array([cluster_colours.get(label, [255, 255, 255]) for label in labels])
    print("Recolouring complete")
    return recoloured

def downsample_pointcloud(xyz, rgb, labels, factor=0.1):
    """Downsample larger clusters more aggressively."""
    print("Downsampling point cloud")
    unique_labels = np.unique(labels)
    downsampled_xyz, downsampled_rgb = [], []
    for label in unique_labels:
        print(f"Downsampling cluster {label} of {len(unique_labels)}")
        if label == -1:
            continue  # Skip noise
        mask = labels == label
        cluster_points = xyz[mask]
        cluster_colours = rgb[mask]
        sample_size = max(1, int(len(cluster_points) * factor))
        sampled_indices = np.random.choice(len(cluster_points), sample_size, replace=False)
        downsampled_xyz.append(cluster_points[sampled_indices])
        downsampled_rgb.append(cluster_colours[sampled_indices])
    return np.vstack(downsampled_xyz), np.vstack(downsampled_rgb)

def save_laz(file_path, xyz, rgb, ref_las):
    """Save point cloud to a new .laz file."""
    header = ref_las.header
    las = laspy.LasData(header)

    # Assigning scaled coordinates
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    # Assigning RGB values
    las.red = rgb[:, 0]
    las.green = rgb[:, 1]
    las.blue = rgb[:, 2]

    # Writing to a new file
    las.write(file_path)

def main(input_file, output_downsampled, output_recoloured):
    """Load, process and save a point cloud."""
    print("Start time: ", datetime.datetime.now())
    xyz, rgb, ref_las = load_laz(input_file)
    labels = cluster_points(xyz, rgb)
    recoloured_rgb = recolour_clusters(labels, rgb)
    downsampled_xyz, downsampled_rgb = downsample_pointcloud(xyz, rgb, labels)
    save_laz(output_recoloured, xyz, recoloured_rgb, ref_las)
    save_laz(output_downsampled, downsampled_xyz, downsampled_rgb, ref_las)
    print("Processing complete!")
    print("End time: ", datetime.datetime.now())

# Directory path
local_path = os.path.dirname(os.path.abspath(__file__))
main(f"{local_path}/original_small.laz", f"{local_path}/downsampled.laz", f"{local_path}/recoloured.laz")