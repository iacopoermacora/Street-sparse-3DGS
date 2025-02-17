import numpy as np
import laspy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import random
import os

def load_laz(file_path):
    """Load a .laz point cloud file."""
    print(f"Loading {file_path}")
    with laspy.open(file_path) as las:
        points = las.read()
    print(f"Loaded {len(points)} points from {file_path}")
    xyz = np.vstack([points.x, points.y, points.z]).T
    rgb = np.vstack([points.red, points.green, points.blue]).T
    print("Loaded xyz and rgb")
    return xyz, rgb, points

def cluster_by_color(rgb, eps=10, min_samples=50):
    """Perform DBSCAN clustering based on color similarity."""
    print("Clustering by color")
    scaled_rgb = StandardScaler().fit_transform(rgb)
    color_clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_rgb)
    print("Color clustering complete")
    return color_clustering.labels_

def cluster_spatially(xyz, labels, eps=0.1, min_samples=10):
    """Perform spatial clustering on each large color cluster."""
    print("Clustering spatially")
    unique_labels = np.unique(labels)
    final_labels = -np.ones_like(labels)  # Initialize with noise
    max_label = 0
    for color_label in unique_labels:
        print(f"Clustering color {color_label} of {len(unique_labels)}")
        mask = labels == color_label
        if np.sum(mask) < 50:
            continue  # Skip small clusters
        features = StandardScaler().fit_transform(xyz[mask])
        spatial_clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
        cluster_labels = spatial_clustering.labels_
        cluster_labels[cluster_labels != -1] += max_label  # Shift cluster IDs
        final_labels[mask] = cluster_labels
        max_label = np.max(final_labels) + 1
    return final_labels

def recolour_clusters(labels, rgb):
    """Assign random colours to clusters for visualisation."""
    print("Recolouring clusters")
    unique_labels = np.unique(labels)
    cluster_colours = {label: np.random.randint(0, 255, 3) for label in unique_labels if label != -1}
    recoloured = np.array([cluster_colours.get(label, [255, 255, 255]) for label in labels])
    return recoloured

def downsample_pointcloud(xyz, rgb, labels, factor=0.1):
    """Downsample larger clusters more aggressively."""
    print("Downsampling point cloud")
    unique_labels = np.unique(labels)
    downsampled_xyz, downsampled_rgb = [], []
    for label in unique_labels:
        print(f"Downsampling cluster {label} of {len(unique_labels)}")
        mask = labels == label
        cluster_points = xyz[mask]
        cluster_colours = rgb[mask]
        sample_size = max(1, int(len(cluster_points) * factor)) if label != -1 else int(len(cluster_points) * 0.05)
        sampled_indices = np.random.choice(len(cluster_points), sample_size, replace=False)
        downsampled_xyz.append(cluster_points[sampled_indices])
        downsampled_rgb.append(cluster_colours[sampled_indices])
    return np.vstack(downsampled_xyz), np.vstack(downsampled_rgb)

def save_laz(file_path, xyz, rgb, ref_las):
    """Save point cloud to a new .laz file."""
    print("Saving .laz file")
    header = ref_las.header
    las = laspy.LasData(header)
    las.points = laspy.ScaleAwarePointRecord.empty(len(xyz), header.point_format)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.red = rgb[:, 0].astype(np.uint16)
    las.green = rgb[:, 1].astype(np.uint16)
    las.blue = rgb[:, 2].astype(np.uint16)
    las.write(file_path)

def main(input_file, output_downsampled, output_recoloured):
    xyz, rgb, ref_las = load_laz(input_file)
    color_labels = cluster_by_color(rgb)
    spatial_labels = cluster_spatially(xyz, color_labels)
    recoloured_rgb = recolour_clusters(spatial_labels, rgb)
    downsampled_xyz, downsampled_rgb = downsample_pointcloud(xyz, rgb, spatial_labels)
    save_laz(output_recoloured, xyz, recoloured_rgb, ref_las)
    save_laz(output_downsampled, downsampled_xyz, downsampled_rgb, ref_las)
    print("Processing complete!")

# Directory path
local_path = os.path.dirname(os.path.abspath(__file__))
main(f"{local_path}/original_small.laz", f"{local_path}/downsampled.laz", f"{local_path}/recoloured.laz")