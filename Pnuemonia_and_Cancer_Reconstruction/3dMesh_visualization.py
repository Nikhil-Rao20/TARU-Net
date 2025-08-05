"""
3D Mesh Visualization using Plotly and Open3D.

This script loads a 3D mesh from a file (e.g., STL, OBJ, PLY) and
creates an interactive 3D visualization using Plotly.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

import os
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from typing import Optional

# =============================================================================
# 2. CONSTANTS
# =============================================================================

# Define the path to the mesh file you want to visualize.
# This makes it easy to change the input file without altering the code logic.
INPUT_MESH_PATH = '/content/Inferenced_lung_003.stl'


# =============================================================================
# 3. VISUALIZATION FUNCTION
# =============================================================================

def visualize_mesh_with_plotly(mesh_path: str):
    """
    Loads a 3D mesh file and displays it in an interactive Plotly figure.

    Args:
        mesh_path (str): The path to the 3D mesh file.
                         Supported formats include .stl, .ply, .obj, etc.
    """
    # --- 1. Validate and Load Mesh ---
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at '{mesh_path}'")
        return

    print(f"Loading mesh from: {mesh_path}")
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    except Exception as e:
        print(f"Error: Failed to read mesh file. {e}")
        return

    if not mesh.has_vertices():
        print("Error: The loaded mesh is empty or invalid.")
        return

    # --- 2. Extract Mesh Data ---
    # Convert Open3D data structures to NumPy arrays for Plotly.
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # --- 3. Create Plotly Figure ---
    # Define the core mesh data for the plot.
    mesh_data_args = {
        'x': vertices[:, 0],
        'y': vertices[:, 1],
        'z': vertices[:, 2],
        'i': triangles[:, 0],
        'j': triangles[:, 1],
        'k': triangles[:, 2],
        'opacity': 1.0,
        'name': os.path.basename(mesh_path), # Add a name for the legend.
        'showscale': False # Hide color bar if no specific intensity is mapped.
    }

    # Add vertex colors only if they are present in the mesh file.
    # STL files typically do not contain color information.
    if mesh.has_vertex_colors():
        print("Vertex colors found in mesh. Applying them to the plot.")
        vertex_colors_scaled = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
        mesh_data_args['vertexcolor'] = [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in vertex_colors_scaled]
    else:
        print("No vertex colors found. Using a default color for the mesh.")
        mesh_data_args['color'] = 'lightblue' # Assign a default color.


    fig = go.Figure(data=[go.Mesh3d(**mesh_data_args)])

    # --- 4. Customize Layout and Display ---
    fig.update_layout(
        title=f"Interactive 3D Visualization of {os.path.basename(mesh_path)}",
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
            aspectratio=dict(x=1, y=1, z=1), # Use 'data' for true scale, 'cube' for cube
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40) # Minimize margins
    )

    print("Showing interactive plot...")
    fig.show()


# =============================================================================
# 4. MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # This block ensures the code runs only when the script is executed directly.
    visualize_mesh_with_plotly(INPUT_MESH_PATH)