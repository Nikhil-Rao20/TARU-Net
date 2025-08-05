import os
import open3d as o3d
import numpy as np
from scipy.interpolate import RBFInterpolator

def remove_background(mesh, min_component_size=1000):
    """ Remove unwanted background surfaces based on connected components. """
    if not mesh.has_vertices():
        print("Warning: Mesh has no vertices!")
        return mesh
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_n_triangles = np.array(cluster_n_triangles)
    largest_cluster_idx = np.argmax(cluster_n_triangles)
    filtered_triangles = [i for i, c in enumerate(triangle_clusters) if c == largest_cluster_idx]
    if len(filtered_triangles) < min_component_size:
        print("Warning: The detected component is too small, check the input!")
        return mesh
    mesh_filtered = mesh.select_by_index(filtered_triangles, invert=False)
    return mesh_filtered


def set_mesh_color(mesh, color):
    """ Set uniform color for the mesh. """
    if not mesh.has_vertices():
        print("Mesh has no vertices to color.")
        return
    colors = np.tile(color, (len(mesh.vertices), 1))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)


def apply_rbf_surface_fitting(pcd):
    """ Fit an implicit RBF surface to smooth and reconstruct the lung shape. """
    points = np.asarray(pcd.points)
    num_samples = min(len(points), 5000)
    sampled_indices = np.random.choice(len(points), num_samples, replace=False)
    sampled_points = points[sampled_indices]
    rbf = RBFInterpolator(sampled_points, np.zeros(len(sampled_points)), kernel='thin_plate_spline')
    smoothed_points = points + 0.01 * np.expand_dims(rbf(points), axis=1)
    pcd.points = o3d.utility.Vector3dVector(smoothed_points)
    return pcd

def process_point_cloud(input_file, output_mesh_file, color=(0.4, 0.4, 0.4), use_bpa=True):
    print(f"Processing: {input_file}")
    pcd = o3d.io.read_point_cloud(input_file)
    # Reduce point cloud density
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    # Remove noise
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd = apply_rbf_surface_fitting(pcd)
    print("RBF smoothing applied.")

    if use_bpa:
        # Ball Pivoting Algorithm (BPA) for meshing
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([avg_dist * 1.5, avg_dist * 3.0])
        )
        print("Ball Pivoting reconstruction complete.")
    else:
        # Poisson Reconstruction (only if BPA fails)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)
        mesh = remove_background(mesh, density_threshold=0.1)
        print("Poisson reconstruction complete with background removal.")

    if not mesh.has_vertices():
        print(f"No vertices found in the generated mesh for {input_file}. Skipping.")
        return

    mesh.compute_vertex_normals()
    set_mesh_color(mesh, color)

    o3d.io.write_triangle_mesh(output_mesh_file, mesh, write_vertex_normals=True, write_vertex_colors=True)
    print(f"Processed and saved: {output_mesh_file}")

def visualize_mesh(mesh_file):
    """ Load and visualize the mesh. """
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        print(f"Mesh {mesh_file} has no vertex colors. Visualizing without colors.")
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

def compute_properties(mesh_file):
    """ Compute surface area of the mesh. """
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    return mesh.get_surface_area()
