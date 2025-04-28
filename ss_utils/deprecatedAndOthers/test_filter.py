import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse

def load_mesh(mesh_path):
    """Load a mesh from a PLY file."""
    print(f"Loading mesh from {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    if not mesh.has_triangles():
        raise ValueError("The loaded mesh has no triangles.")
    
    # Ensure the mesh has vertex normals for distance computation
    if not mesh.has_vertex_normals():
        print("Computing mesh vertex normals...")
        mesh.compute_vertex_normals()
        
    return mesh

def load_point_cloud(pcd_path):
    """Load a point cloud from a PLY file."""
    print(f"Loading point cloud from {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    if not pcd.has_points():
        raise ValueError("The loaded point cloud has no points.")
        
    return pcd

def filter_points_by_distance(pcd, mesh, max_distance=0.1):
    """
    Filter points in the point cloud that are within max_distance from the mesh.
    
    Args:
        pcd: The point cloud (o3d.geometry.PointCloud)
        mesh: The reference mesh (o3d.geometry.TriangleMesh)
        max_distance: Maximum distance threshold in meters (0.1 = 10 cm)
        
    Returns:
        filtered_pcd: Point cloud containing only points within the threshold
    """
    print(f"Filtering points (max distance: {max_distance*100:.1f} cm)...")
    
    # Convert point cloud to numpy array for processing
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    
    # Create a scene with the mesh for distance queries
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
    
    # Compute the distance from each point to the mesh
    # Using unsigned distance (absolute value)
    query_points = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    distances = scene.compute_distance(query_points).numpy()
    
    # Filter points based on distance threshold
    mask = distances <= max_distance
    filtered_points = points[mask]
    
    # Create a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # Copy colors and normals if they exist
    if colors is not None:
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    if normals is not None:
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals[mask])
    
    print(f"Original point count: {len(points)}")
    print(f"Filtered point count: {len(filtered_points)}")
    print(f"Removed {len(points) - len(filtered_points)} points ({(1 - len(filtered_points)/len(points))*100:.2f}%)")
    
    return filtered_pcd

def process_in_batches(pcd, mesh, max_distance=0.1, batch_size=100000):
    """
    Process large point clouds in batches to avoid memory issues.
    
    Args:
        pcd: The point cloud (o3d.geometry.PointCloud)
        mesh: The reference mesh (o3d.geometry.TriangleMesh)
        max_distance: Maximum distance threshold in meters (0.1 = 10 cm)
        batch_size: Number of points to process in each batch
        
    Returns:
        filtered_pcd: Point cloud containing only points within the threshold
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    
    # Create a scene with the mesh for distance queries
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
    
    # Calculate number of batches
    n_points = len(points)
    n_batches = int(np.ceil(n_points / batch_size))
    
    # Lists to store filtered data
    filtered_points = []
    filtered_colors = [] if colors is not None else None
    filtered_normals = [] if normals is not None else None
    
    print(f"Processing {n_points} points in {n_batches} batches...")
    
    # Process each batch
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_points)
        
        # Get current batch
        batch_points = points[start_idx:end_idx]
        
        # Compute distances for the current batch
        query_points = o3d.core.Tensor(batch_points, dtype=o3d.core.Dtype.Float32)
        batch_distances = scene.compute_distance(query_points).numpy()
        
        # Filter based on distance threshold
        batch_mask = batch_distances <= max_distance
        
        # Append filtered points
        filtered_points.append(batch_points[batch_mask])
        
        # Append filtered colors and normals if they exist
        if colors is not None:
            batch_colors = colors[start_idx:end_idx]
            filtered_colors.append(batch_colors[batch_mask])
        
        if normals is not None:
            batch_normals = normals[start_idx:end_idx]
            filtered_normals.append(batch_normals[batch_mask])
    
    # Combine all filtered batches
    filtered_points = np.vstack(filtered_points) if filtered_points else np.array([])
    
    if colors is not None:
        filtered_colors = np.vstack(filtered_colors) if filtered_colors else np.array([])
    
    if normals is not None:
        filtered_normals = np.vstack(filtered_normals) if filtered_normals else np.array([])
    
    # Create a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # Copy colors and normals if they exist
    if colors is not None:
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    if normals is not None:
        filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
    
    print(f"Original point count: {n_points}")
    print(f"Filtered point count: {len(filtered_points)}")
    print(f"Removed {n_points - len(filtered_points)} points ({(1 - len(filtered_points)/n_points)*100:.2f}%)")
    
    return filtered_pcd

def main():
    parser = argparse.ArgumentParser(description="Filter point cloud by distance to mesh")
    parser.add_argument("--mesh", required=True, help="Path to the reference mesh PLY file")
    parser.add_argument("--pointcloud", required=True, help="Path to the point cloud PLY file")
    parser.add_argument("--output", required=True, help="Path to save the filtered point cloud")
    parser.add_argument("--distance", type=float, default=0.1, help="Maximum distance threshold in meters (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=100000, help="Batch size for processing large point clouds")
    
    args = parser.parse_args()
    
    # Load mesh and point cloud
    mesh = load_mesh(args.mesh)
    pcd = load_point_cloud(args.pointcloud)
    
    # Filter points by distance
    filtered_pcd = process_in_batches(pcd, mesh, max_distance=args.distance, batch_size=args.batch_size)
    
    # Save the filtered point cloud
    print(f"Saving filtered point cloud to {args.output}...")
    o3d.io.write_point_cloud(args.output, filtered_pcd)
    print("Done!")

if __name__ == "__main__":
    main()