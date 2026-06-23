"""Render an Azure Kinect MKV's colored point cloud in depth camera space and
overlay a PLY mesh on top of it.

The point cloud is built from the first frame's depth image (native depth
camera space) and colored using the color image transformed into depth space.
The PLY mesh is expected to already live in depth camera space (e.g. a mesh
exported by the registration/overlay scripts). If it does not, set
MESH_TRANSFORM to the 4x4 matrix that maps the mesh into depth camera space.
"""

import cv2
import numpy as np
import open3d as o3d
from pyk4a import PyK4APlayback, ImageFormat


# --- Config ---------------------------------------------------------------
MKV_PATH = "/home/connorscomputer/Desktop/RVGGhxD2_20250930_123217.mkv"
# PLY mesh already in depth camera space.
PLY_PATH = "/home/connorscomputer/Desktop/RVGGhxD2_20250930_123217_transformed_mesh_final.ply"
# Seconds to seek into the recording before grabbing a frame.
OFFSET_SEC = 0.0
# Optional 4x4 transform applied to the mesh (identity = mesh already in depth space).
MESH_TRANSFORM = np.eye(4)
# Paint the mesh a solid color instead of using its own vertex colors.
PAINT_MESH = False
MESH_COLOR = [1.0, 1.0, 0.0]  # yellow


def load_point_cloud_from_capture(capture):
    """Build a colored Open3D point cloud (depth camera space) from a capture."""
    # Decode the JPEG color image and convert to BGRA so pyk4a can transform it.
    capture._color = cv2.cvtColor(
        cv2.imdecode(capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
    capture._color_format = ImageFormat.COLOR_BGRA32

    # Color image warped into the depth camera frame (per-depth-pixel color).
    transformed_color = capture.transformed_color
    if transformed_color is not None:
        colors = transformed_color[..., (2, 1, 0)].reshape(
            (-1, 3))  # BGRA->RGB
    else:
        print("Warning: no transformed color, using gray")
        colors = None

    # Point cloud from depth, in native depth camera space (mm).
    points = capture.depth_point_cloud.reshape((-1, 3)).astype("float64")
    if colors is None:
        colors = np.full((points.shape[0], 3), 128, dtype=np.uint8)

    # Drop invalid (zero-depth) points.
    valid = points[:, 2] > 0
    points = points[valid]
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector((colors / 255).astype("float64"))
    return pcd


def main():
    print(f"Opening MKV: {MKV_PATH}")
    playback = PyK4APlayback(MKV_PATH)
    playback.open()
    print(f"Record length: {playback.length / 1e6:0.2f} sec")

    if OFFSET_SEC != 0.0:
        playback.seek(int(OFFSET_SEC * 1e6))

    capture = playback.get_next_capture()
    if capture.color is None or capture.depth is None:
        print("Error: first capture has no color or depth")
        playback.close()
        return

    pcd = load_point_cloud_from_capture(capture)
    print(f"Point cloud: {len(pcd.points)} points")
    playback.close()

    print(f"Loading mesh: {PLY_PATH}")
    mesh = o3d.io.read_triangle_mesh(PLY_PATH)
    if not mesh.has_vertices():
        print("Error: mesh has no vertices")
        return
    mesh.compute_vertex_normals()
    if not np.allclose(MESH_TRANSFORM, np.eye(4)):
        mesh.transform(MESH_TRANSFORM)
    if PAINT_MESH or not mesh.has_vertex_colors():
        mesh.paint_uniform_color(MESH_COLOR)
    print(
        f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    # Depth camera coordinate frame at the origin.
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100.0, origin=[0, 0, 0])

    print("\nVisualizing (Q or ESC to close)...")
    o3d.visualization.draw_geometries(
        [pcd, mesh, camera_frame],
        window_name="Point Cloud + Mesh (Depth Camera Space)",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
