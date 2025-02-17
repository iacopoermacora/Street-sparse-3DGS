import laspy
from collections import Counter
import numpy as np
import statistics

def get_top_colors(laz_file, top_n=500, output_file="top_colors.txt"):
    # Read the LAZ file
    with laspy.open(laz_file) as file:
        las = file.read()

    print("File read")

    # Extract RGB values (assuming they are stored as uint16, normalize to 8-bit)
    red = (las.red / 256).astype(np.uint8)
    green = (las.green / 256).astype(np.uint8)
    blue = (las.blue / 256).astype(np.uint8)
    
    # Extract XYZ coordinates
    x_coords = las.x
    y_coords = las.y
    z_coords = las.z

    print("Creating color tuples...")
    
    # Create color tuples
    colors = list(zip(red, green, blue))
    
    # Count occurrences of each color and store coordinates
    color_counts = Counter(colors)
    color_positions = {color: [] for color in color_counts}
    
    for i, color in enumerate(colors):
        color_positions[color].append((x_coords[i], y_coords[i], z_coords[i]))
    
    print("Retrieving top colors...")

    # Get the top N colors
    top_colors = color_counts.most_common(top_n)
    
    print("Writing to output file...")

    # Write to output file
    with open(output_file, "w") as f:
        for color, count in top_colors:
            positions = np.array(color_positions[color])
            avg_position = np.mean(positions, axis=0)
            median_position = np.median(positions, axis=0)
            mode_position = [
                statistics.mode(positions[:, 0]),
                statistics.mode(positions[:, 1]),
                statistics.mode(positions[:, 2])
            ]
            
            f.write(f"{color}: {count}\n")
            f.write(f"  Avg Position: {avg_position}\n")
            f.write(f"  Median Position: {median_position}\n")
            f.write(f"  Mode Position: {mode_position}\n\n")
    
    return top_colors

# Example usage
laz_file_path = "/home/local/CYCLOMEDIA001/iermacora/Street-sparse-3DGS/ss_raw_images/LiDAR/filtered_2733_9116.laz"  # Replace with your file path
top_colors = get_top_colors(laz_file_path)