"""
Utility functions for 3D point cloud and mesh processing using Open3D.

This module provides a pipeline for:
1.  Reading point cloud data.
2.  Cleaning and preprocessing the point cloud (down-sampling, outlier removal).
3.  Smoothing the point cloud surface using RBF interpolation.
4.  Reconstructing a 3D mesh from the point cloud using Ball Pivoting or Poisson.
5.  Post-processing the mesh (cleaning, coloring).
6.  Visualizing and computing properties of the final mesh.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

import logging
import os
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.interpolate import RBFInterpolator

# =============================================================================
# 2. INITIAL CONFIGURATION
# =============================================================================

# Configure logging for clear and controllable output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# --- Default Algorithm Parameters (as constants for clarity) ---
DEFAULT_VOXEL_SIZE = 0.1
DEFAULT_OUTLIER_NEIGHBORS = 10
DEFAULT_OUTLIER_STD_RATIO = 1.0
DEFAULT_NORMAL_RADIUS = 0.1
DEFAULT_NORMAL_MAX_NN = 30
DEFAULT_POISSON_DEPTH = 8
DEFAULT_POISSON_DENSITY_QUANTILE = 0.01
DEFAULT_RBF_SAMPLE_COUNT = 2000


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def keep_largest_mesh_component(mesh: o3d.geometry.TriangleMesh, min_component_triangles: int = 1000) -> o3d.geometry.TriangleMesh:
    """
    Isolates the largest connected component of a mesh.

    This is useful for removing small, disconnected "island" artifacts from a mesh.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        min_component_triangles (int): The minimum number of triangles for the
                                       largest component to be considered valid.

    Returns:
        o3d.geometry.TriangleMesh: The cleaned mesh containing only the largest component.
    """
    if not mesh.has_triangles():
        logging.warning("Mesh has no triangles; cannot find connected components.")
        return mesh

    # Find all connected components (clusters of triangles)
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    
    if not cluster_n_triangles:
        logging.warning("Could not find any triangle clusters in the mesh.")
        return mesh
        
    largest_cluster_idx = np.argmax(cluster_n_triangles)
    
    if cluster_n_triangles[largest_cluster_idx] < min_component_triangles:
        logging.warning(
            f"Largest component ({cluster_n_triangles[largest_cluster_idx]} triangles) "
            f"is smaller than the minimum required size ({min_component_triangles}). "
            "Returning original mesh to avoid removing main object."
        )
        return mesh

    # Create a mask to select triangles belonging to the largest cluster
    triangles_to_keep = [i for i, cluster_id in enumerate(triangle_clusters) if cluster_id == largest_cluster_idx]
    
    cleaned_mesh = mesh.select_by_index(triangles_to_keep)
    logging.info(f"Kept largest component with {len(cleaned_mesh.triangles)} triangles.")
    return cleaned_mesh


def smooth_point_cloud_rbf(pcd: o3d.geometry.PointCloud, num_sample_points: int = DEFAULT_RBF_SAMPLE_COUNT) -> o3d.geometry.PointCloud:
    """
    Fits an implicit RBF surface to smooth and regularize the point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        num_sample_points (int): The number of points to sample for the RBF fit.

    Returns:
        o3d.geometry.PointCloud: A new point cloud with smoothed point positions.
    """
    points = np.asarray(pcd.points)
    if len(points) < num_sample_points:
        logging.warning("Point cloud has fewer points than RBF sample count. Using all points.")
        num_sample_points = len(points)

    # Randomly sample points to build the RBF interpolator
    sampled_indices = np.random.choice(len(points), num_sample_points, replace=False)
    sampled_points = points[sampled_indices]

    # Use RBF to define an implicit function where the surface is at f(x,y,z) = 0
    rbf = RBFInterpolator(sampled_points, np.zeros(num_sample_points), kernel='thin_plate_spline', smoothing=0.1)
    
    # Evaluate the implicit function at all point locations
    implicit_values = rbf(points)
    
    # Move points towards the zero-level surface along their normal vectors
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    smoothed_points = points - normals * implicit_values[:, np.newaxis]

    smoothed_pcd = o3d.geometry.PointCloud()
    smoothed_pcd.points = o3d.utility.Vector3dVector(smoothed_points)
    logging.info(f"Applied RBF smoothing to {len(points)} points.")
    return smoothed_pcd


def process_point_cloud(
    input_xyz_path: str,
    output_mesh_path: str,
    color: Tuple[float, float, float] = (0.4, 0.4, 0.4),
    use_bpa: bool = True
):
    """
    Full pipeline to process a point cloud file into a clean 3D mesh.

    Args:
        input_xyz_path (str): Path to the input point cloud file (.xyz, .ply).
        output_mesh_path (str): Path to save the final mesh file (.obj, .ply).
        color (Tuple[float, float, float]): RGB color tuple for the final mesh.
        use_bpa (bool): If True, use Ball Pivoting Algorithm. If False, use Poisson.
    """
    logging.info(f"Starting processing for: {input_xyz_path}")
    try:
        pcd = o3d.io.read_point_cloud(input_xyz_path)
    except Exception as e:
        logging.error(f"Failed to read point cloud file: {e}")
        return

    if not pcd.has_points():
        logging.error("Input point cloud is empty. Aborting.")
        return

    # 1. Preprocessing: Down-sample and remove outliers
    pcd = pcd.voxel_down_sample(voxel_size=DEFAULT_VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=DEFAULT_OUTLIER_NEIGHBORS, std_ratio=DEFAULT_OUTLIER_STD_RATIO
    )
    logging.info(f"Preprocessing complete. Points after cleaning: {len(pcd.points)}")

    # 2. RBF Smoothing
    pcd = smooth_point_cloud_rbf(pcd)
    
    # 3. Normal Estimation (critical for reconstruction)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=DEFAULT_NORMAL_RADIUS, max_nn=DEFAULT_NORMAL_MAX_NN)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15) # Orient normals outward

    # 4. Mesh Reconstruction
    if use_bpa:
        logging.info("Reconstructing mesh using Ball Pivoting Algorithm (BPA)...")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist, avg_dist * 2]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    else:
        logging.info("Reconstructing mesh using Screened Poisson Surface Reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=DEFAULT_POISSON_DEPTH, linear_fit=True
        )
        # Clean Poisson mesh by removing low-density vertices
        density_threshold = np.quantile(densities, DEFAULT_POISSON_DENSITY_QUANTILE)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

    if not mesh.has_vertices():
        logging.error("Mesh reconstruction failed. The resulting mesh has no vertices.")
        return
        
    logging.info("Initial mesh reconstruction complete.")

    # 5. Post-processing: Clean up artifacts
    mesh = keep_largest_mesh_component(mesh)
    mesh.compute_vertex_normals()

    # 6. Finalize and Save
    mesh.paint_uniform_color(color)
    try:
        o3d.io.write_triangle_mesh(output_mesh_path, mesh, write_vertex_colors=True)
        logging.info(f"Successfully saved final mesh to: {output_mesh_path}")
    except Exception as e:
        logging.error(f"Failed to write mesh file: {e}")


def visualize_mesh(mesh_path: str):
    """Loads and visualizes a mesh file using the Open3D visualizer."""
    logging.info(f"Attempting to visualize: {mesh_path}")
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    except Exception as e:
        logging.error(f"Could not read mesh file for visualization: {e}")
        return
        
    if not mesh.has_vertices():
        logging.error("Cannot visualize an empty mesh.")
        return
        
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name=f"Visualization of {os.path.basename(mesh_path)}")


def compute_surface_area(mesh_path: str) -> Optional[float]:
    """Computes the surface area of a mesh."""
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh.has_vertices():
            return mesh.get_surface_area()
        else:
            logging.warning("Cannot compute surface area of an empty mesh.")
            return None
    except Exception as e:
        logging.error(f"Could not read mesh file to compute properties: {e}")
        return None

# =============================================================================
# 4. DEMONSTRATION BLOCK
# =============================================================================

if __name__ == "__main__":
    logging.info("Running demonstration of the mesh processing module...")

    # Create a dummy point cloud file for demonstration purposes
    demo_dir = "demo_files"
    os.makedirs(demo_dir, exist_ok=True)
    demo_xyz_path = os.path.join(demo_dir, "sample_sphere.xyz")
    demo_mesh_path = os.path.join(demo_dir, "reconstructed_sphere.obj")

    # Generate points for a noisy sphere
    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2 * np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    points += 0.05 * np.random.randn(*points.shape) # Add noise
    np.savetxt(demo_xyz_path, points)

    logging.info(f"Generated a sample point cloud at: {demo_xyz_path}")

    # --- Run the full processing pipeline ---
    process_point_cloud(
        input_xyz_path=demo_xyz_path,
        output_mesh_path=demo_mesh_path,
        color=(0.8, 0.2, 0.2), # A nice red color
        use_bpa=True
    )

    # --- Compute properties of the output ---
    surface_area = compute_surface_area(demo_mesh_path)
    if surface_area is not None:
        logging.info(f"Computed surface area of the sphere: {surface_area:.4f}")

    # --- Visualize the final result ---
    if os.path.exists(demo_mesh_path):
        visualize_mesh(demo_mesh_path)
    else:
        logging.error("Demonstration failed: Output mesh was not created.")