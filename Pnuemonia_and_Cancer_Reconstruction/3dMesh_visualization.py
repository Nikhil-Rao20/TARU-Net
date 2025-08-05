# Load with Plotly (approximate)
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go

# Save mesh as .ply (which supports vertex colors better for web)
mesh = o3d.io.read_triangle_mesh('/content/Inferenced_lung_003.stl')
o3d.io.write_triangle_mesh("colored_lung_attempt.ply", mesh)

mesh = o3d.io.read_triangle_mesh("colored_lung_attempt.ply")
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
colors = np.asarray(mesh.vertex_colors)

fig = go.Figure(
    data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            vertexcolor=colors,
            opacity=1.0
        )
    ]
)
fig.update_layout(scene=dict(aspectmode="data"))
fig.show()
