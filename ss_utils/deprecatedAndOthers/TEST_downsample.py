import open3d as o3d
import laspy
import numpy as np
import sys
import os

# Assuming downsample_for_max_density function is defined elsewhere
# Replace this with your actual downsample_for_max_density function
def downsample_for_max_density(pcd, max_density):
    """
    Downsamples a point cloud to ensure its density does not exceed the specified threshold.
    
    Parameters:
    pcd (open3d.geometry.PointCloud): The input point cloud.
    max_density (float): Maximum allowed density in points per cubic unit (consistent with the point cloud's units).
    
    Returns:
    open3d.geometry.PointCloud: Downsampled point cloud with density <= max_density.
    """
    # Calculate the voxel size required to achieve the maximum density
    voxel_size = (1.0 / max_density) ** (1/3)
    
    # Perform voxel downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    
    return downsampled_pcd

def laz_to_o3d_pointcloud(laz_filepath):
    """
    Reads a LAZ file using laspy and converts it to an Open3D PointCloud.
    Compatible with laspy 2.x

    Args:
        laz_filepath (str): Path to the LAZ file.

    Returns:
        open3d.geometry.PointCloud: Open3D PointCloud object, or None if loading fails.
    """
    try:
        # Read the LAZ file with laspy (using laspy.read for version 2.0 compatibility)
        lasf = laspy.read(laz_filepath)

        points = np.vstack((lasf.x, lasf.y, lasf.z)).transpose() # Get X, Y, Z coordinates as a NumPy array

        # Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points) # Assign the points

        # --- Handle colors if available (Optional) - laspy 2.x compatible ---
        has_color = False
        color_channels = {}
        for dimension in lasf.header.point_format.dimensions:
            dimension_name = dimension.name.lower() # Convert to lowercase for case-insensitive check
            if dimension_name in ['red', 'green', 'blue']:
                has_color = True
                color_channels[dimension_name] = dimension_name # Store the dimension names

        if has_color and 'red' in color_channels and 'green' in color_channels and 'blue' in color_channels:
            colors = np.vstack((lasf[color_channels['red']], lasf[color_channels['green']], lasf[color_channels['blue']])).transpose()
            colors = colors.astype(np.float32) / 65535.0  # Scale color values to [0, 1] if necessary (adjust scaling based on your LAZ file's color format)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    except Exception as e:
        print(f"Error reading LAZ file '{laz_filepath}' with laspy: {e}")
        return None

if __name__ == "__main__":
    # Read LiDAR points once
    print("Reading LiDAR points")
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    laz_directory = f"{base_path}/ss_raw_images/LiDAR"

    if not os.path.isdir(laz_directory):
        print(f"Error: Directory '{laz_directory}' not found.")
        sys.exit(1)

    laz_files = []
    for filename in os.listdir(laz_directory):
        if filename.lower().endswith(".laz"):
            filepath = os.path.join(laz_directory, filename)
            laz_files.append(filepath)

    if not laz_files:
        print(f"Warning: No .laz files found in directory '{laz_directory}'.")
        sys.exit(1)

    combined_pcd = o3d.geometry.PointCloud() # Initialize an empty point cloud to store combined result

    for laz_file in laz_files:
        if not os.path.exists(laz_file): # Redundant check, but good practice
            print(f"Error: File '{laz_file}' not found (this should not happen).")
            continue # Should not reach here, but just in case

        try:
            print(f"Reading point cloud from: {laz_file}")
            pcd = laz_to_o3d_pointcloud(laz_file) # Use the conversion function
            print(f"Point cloud loaded with {len(pcd.points)} points.")

            # Define the maximum density (points per cubic unit)
            max_density = 8000  # You can adjust this value

            # Downsample the point cloud
            print("Downsampling point cloud...")
            downsampled_pcd = downsample_for_max_density(pcd, max_density)
            print(f"Point cloud downsampled. Points after downsampling: {len(downsampled_pcd.points)}")

            # Combine the downsampled point cloud with the combined_pcd
            combined_pcd = combined_pcd + downsampled_pcd  # Using the + operator to combine point clouds

        except Exception as e:
            print(f"Error processing file '{laz_file}': {e}")

    if len(combined_pcd.points) > 0: # Only save if there are points in the combined point cloud
        output_file = "combined_downsampled.pcd" # Define the output filename for the combined point cloud

        # Save the combined result
        print(f"Saving combined downsampled point cloud to: {output_file}")
        o3d.io.write_point_cloud(output_file, combined_pcd)
        print(f"Combined downsampled point cloud saved to: {output_file}")
    else:
        print("No point clouds were processed successfully or the combined point cloud is empty. No output file saved.")

    print("Finished processing all files and combining results.")