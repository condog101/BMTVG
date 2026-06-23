from argparse import ArgumentParser
import cv2
import numpy as np
from typing import Optional, Tuple
from pyk4a import PyK4APlayback, ImageFormat, CalibrationType
import json
import matplotlib.pyplot as plt
import open3d as o3d


def load_intrinsics_from_calibration(calibration, camera_type=CalibrationType.DEPTH):
    """Extract intrinsics from PyK4A calibration."""
    K = calibration.get_camera_matrix(camera_type)
    dist = calibration.get_distortion_coefficients(camera_type)
    return K.astype(np.float32), dist.astype(np.float32)


def get_color_to_depth_transform(calibration):
    """
    Get the 4x4 transformation matrix from color camera space to depth camera space.
    """
    rotation, translation = calibration.get_extrinsic_parameters(
        CalibrationType.COLOR, CalibrationType.DEPTH
    )
    # translation is returned in meters, convert to mm for consistency
    translation_mm = translation.flatten() * 1000.0

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = translation_mm
    return T


def rvec_tvec_to_T(rvec, tvec):
    """Convert rotation vector and translation to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


class BoardTracker:
    def __init__(self, K: np.ndarray, dist: np.ndarray, marker_size_mm=30.0, aruco_dict_name="DICT_6X6_50"):
        self.K = K.astype(np.float32)
        self.dist = dist.astype(np.float32)
        self.marker_size_mm = marker_size_mm
        self.marker_stl = "/home/connorscomputer/Desktop/hex30_fusion_coordinates.stl"

        dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        }
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            dict_map.get(aruco_dict_name, cv2.aruco.DICT_6X6_50))
        self.detector_params = cv2.aruco.DetectorParameters(
        )
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

        # self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.detector_params)

        self.tools = {}
        self._last_ids = None
        self._last_corners = None

    def initialize_from_config(self, board_config_json_path: str):
        with open(board_config_json_path, "r") as f:
            cfg = json.load(f)

        self.tools.clear()
        for tool in cfg.get("toolList", []):
            name = tool["id"]
            ids = tool["marker_ids"]
            corners = tool["marker_corners_m"]
            per_marker = {}
            for mid, m_corners in zip(ids, corners):
                # Keep in mm to match Azure Kinect point cloud units
                arr = np.array(m_corners, dtype=np.float32)
                per_marker[int(mid)] = arr
            self.tools[name] = per_marker

    def detect(self, image: np.ndarray):
        corners, ids, _ = self.detector.detectMarkers(image)
        self._last_corners = corners
        self._last_ids = ids
        return corners, ids

    def _collect_tool_correspondences(self, tool_name):
        marker_map = self.tools[tool_name]
        if self._last_ids is None or len(self._last_ids) == 0:
            return None, None

        id_to_idx = {int(_id): idx for idx, _id in enumerate(
            self._last_ids.flatten())}
        obj_pts, img_pts = [], []

        for mid, obj_corners in marker_map.items():
            if mid in id_to_idx:
                di = id_to_idx[mid]
                img_corners = self._last_corners[di].reshape(-1, 2)
                obj_pts.append(obj_corners)
                img_pts.append(img_corners)

        if not obj_pts:
            return None, None

        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)
        return obj_pts, img_pts

    def RequestPose(self, tool_name: str):
        if tool_name not in self.tools:
            return False, np.eye(4, dtype=np.float32), None, None, {}

        obj_pts, img_pts = self._collect_tool_correspondences(tool_name)
        if obj_pts is None:
            return False, np.eye(4, dtype=np.float32), None, None, {}

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=self.K,
            distCoeffs=self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            confidence=0.99,
            iterationsCount=100
        )
        if not success:
            return False, np.eye(4, dtype=np.float32), None, None, {}

        # Compute projection error metrics
        projected_pts, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, self.K, self.dist)
        projected_pts = projected_pts.reshape(-1, 2)

        # Calculate per-point errors
        errors = np.linalg.norm(img_pts - projected_pts, axis=1)

        metrics = {
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'std_error': float(np.std(errors)),
            'median_error': float(np.median(errors)),
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'num_points': len(obj_pts),
            'num_inliers': int(len(inliers)) if inliers is not None else len(obj_pts),
            'inlier_ratio': float(len(inliers) / len(obj_pts)) if inliers is not None else 1.0
        }

        return True, rvec_tvec_to_T(rvec, tvec), rvec, tvec, metrics

    def GetTrackableNames(self):
        return list(self.tools.keys())

    def annotate(self, image: np.ndarray):
        """Draw detected markers and coordinate axes on image."""
        out = image.copy()

        # Draw detected markers with IDs
        if self._last_ids is not None and len(self._last_ids) > 0:
            cv2.aruco.drawDetectedMarkers(
                out, self._last_corners, self._last_ids)

        # Draw coordinate frame for each detected tool
        for name in self.GetTrackableNames():
            found, T, rvec, tvec, metrics = self.RequestPose(name)

            if found:
                # Draw 3D coordinate axes with swapped red/blue (X=blue, Y=green, Z=red)
                axis_length = self.marker_size_mm * 2
                axis_pts = np.array([
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, axis_length],
                ], dtype=np.float32)
                projected, _ = cv2.projectPoints(
                    axis_pts, rvec, tvec, self.K, self.dist)
                projected = projected.reshape(-1, 2).astype(int)
                origin = tuple(projected[0])
                cv2.line(out, origin, tuple(
                    projected[1]), (255, 0, 0), 3)   # X = blue
                cv2.line(out, origin, tuple(
                    projected[2]), (0, 255, 0), 3)   # Y = green
                cv2.line(out, origin, tuple(
                    projected[3]), (0, 0, 255), 3)   # Z = red

        return out


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def main() -> None:

    filename = "/home/connorscomputer/Desktop/Z4AeN88U_20251204_134045.mkv"
    spine_config = "/home/connorscomputer/recording_scripts/board_config.json"
    # Path to a mesh already in depth camera space (e.g. "/path/to/mesh.ply")
    overlay_mesh_path = "/home/connorscomputer/Desktop/Z4AeN88U_20251204_134045_transformed_mesh_edited.ply"
    offset = 0.0
    sequence_name = filename.split("/")[-1].replace(".mkv", "")

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)

    # Get intrinsics from first capture
    capture = playback.get_next_capture()
    K_color, dist_color = load_intrinsics_from_calibration(
        capture._calibration, CalibrationType.COLOR)

    # Get the extrinsic transform from color camera to depth camera
    T_color_to_depth = get_color_to_depth_transform(capture._calibration)
    print(f"\nColor to Depth extrinsic transform:")
    print(
        f"  Translation (mm): [{T_color_to_depth[0,3]:.2f}, {T_color_to_depth[1,3]:.2f}, {T_color_to_depth[2,3]:.2f}]")

    # Seek to offset if specified
    if offset != 0.0:
        playback.seek(int(offset * 1000000))
        capture = playback.get_next_capture()

    # Initialize tracker with COLOR camera intrinsics for detection and pose estimation
    # We'll chain the transform afterwards to get depth camera space
    tracker = BoardTracker(K_color, dist_color, marker_size_mm=20.0,
                           aruco_dict_name="DICT_6X6_50")
    tracker.initialize_from_config(spine_config)

    print(f"\nTracking tools: {tracker.GetTrackableNames()}")
    print(f"Using 6x6 ArUco markers, 30mm size")
    print(f"Detection in COLOR camera (high-res), transform chained to DEPTH camera space")

    # Process one frame
    if capture.color is not None and capture.depth is not None:
        # Prepare color image for ArUco detection
        capture._color = cv2.cvtColor(cv2.imdecode(
            capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
        capture._color_format = ImageFormat.COLOR_BGRA32

        color_bgr = capture._color[..., (2, 1, 0)]

        # Get color image transformed to depth camera space (for visualization only)
        transformed_color = capture.transformed_color
        if transformed_color is not None:
            transformed_color_bgr = transformed_color[..., (2, 1, 0)]
        else:
            print("Warning: Could not get transformed color image")
            transformed_color_bgr = None

        # Detect markers in the HIGH-RES color image (better detection)
        corners, ids = tracker.detect(color_bgr)

        print(f"\nDetected {len(ids) if ids is not None else 0} markers")
        if ids is not None:
            print(f"Marker IDs: {ids.flatten()}")

        # Annotate the color image
        annotated = tracker.annotate(color_bgr)

        # Check pose for each tool
        for name in tracker.GetTrackableNames():
            found, T_marker_to_color, rvec, tvec, metrics = tracker.RequestPose(
                name)
            if found:
                # Chain transforms: marker -> color camera -> depth camera
                # T_marker_to_depth = T_color_to_depth @ T_marker_to_color
                T_marker_to_depth = T_color_to_depth @ T_marker_to_color.astype(
                    np.float64)

                print(f"\n{name} pose found:")
                print(
                    f"  Position (color camera space): [{T_marker_to_color[0,3]:.1f}, {T_marker_to_color[1,3]:.1f}, {T_marker_to_color[2,3]:.1f}] mm")
                print(
                    f"  Position (depth camera space): [{T_marker_to_depth[0,3]:.1f}, {T_marker_to_depth[1,3]:.1f}, {T_marker_to_depth[2,3]:.1f}] mm")
                print(f"\n  Projection Error Metrics:")
                print(f"    RMSE:          {metrics['rmse']:.3f} pixels")
                print(f"    Mean Error:    {metrics['mean_error']:.3f} pixels")
                print(
                    f"    Median Error:  {metrics['median_error']:.3f} pixels")
                print(f"    Max Error:     {metrics['max_error']:.3f} pixels")
                print(f"    Min Error:     {metrics['min_error']:.3f} pixels")
                print(f"    Std Dev:       {metrics['std_error']:.3f} pixels")
                print(
                    f"    Points Used:   {metrics['num_inliers']}/{metrics['num_points']} (inlier ratio: {metrics['inlier_ratio']:.2%})")

                # Save transform as .npy file (in depth camera space)
                np.save(
                    f"transform_{name}_to_depth_camera_{sequence_name}.npy", T_marker_to_depth)
                print(
                    f"\n  Saved transform for {name} to transform_{name}_to_depth_camera_{sequence_name}.npy")
            else:
                print(f"\n{name}: Not detected")

        # Overlay a mesh (already in depth camera space) onto the color image
        if overlay_mesh_path is not None:
            T_depth_to_color = np.linalg.inv(T_color_to_depth)
            mesh = o3d.io.read_triangle_mesh(overlay_mesh_path)
            if mesh.has_vertices() and mesh.has_vertex_colors():
                vertices = np.asarray(mesh.vertices).astype(np.float64)
                colors = (np.asarray(mesh.vertex_colors)
                          * 255).round().astype(int)
                triangles = np.asarray(mesh.triangles)

                # Segment vertebrae by unique vertex color
                unique_colors, vertex_labels = np.unique(
                    colors, axis=0, return_inverse=True)
                print(f"Found {len(unique_colors)} vertebrae by color")

                # Compute centroid per vertebra
                centroids = np.array([
                    vertices[vertex_labels == i].mean(axis=0)
                    for i in range(len(unique_colors))
                ])

                # Find the spine's long axis (axis with largest spread of centroids)
                spread = centroids.ptp(axis=0)
                long_axis = np.argmax(spread)
                print(
                    f"Spine long axis: {'XYZ'[long_axis]} (spread: {spread[long_axis]:.1f} mm)")

                # Sort vertebrae along the long axis and remove the 2 at opposite ends
                sorted_indices = np.argsort(centroids[:, long_axis])
                remove_indices = [sorted_indices[0], sorted_indices[-1]]
                keep_indices = sorted_indices[1:-1]
                print(
                    f"Removing 2 most distal vertebrae (indices {remove_indices})")

                # Build mask of vertices to keep
                keep_mask = np.isin(vertex_labels, keep_indices)
                keep_triangles_mask = keep_mask[triangles].all(axis=1)
                filtered_triangles = triangles[keep_triangles_mask]

                # Remove small disconnected clusters per vertebra (artefacts)
                tri_labels_raw = vertex_labels[filtered_triangles]
                cleaned_triangles = []
                for vi in keep_indices:
                    vert_tri_mask = np.array([
                        np.bincount(row).argmax() == vi for row in tri_labels_raw
                    ])
                    vert_tris = filtered_triangles[vert_tri_mask]
                    if len(vert_tris) == 0:
                        continue
                    sub_mesh = o3d.geometry.TriangleMesh()
                    sub_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    sub_mesh.triangles = o3d.utility.Vector3iVector(vert_tris)
                    cluster_ids, cluster_counts, _ = sub_mesh.cluster_connected_triangles()
                    cluster_ids = np.asarray(cluster_ids)
                    cluster_counts = np.asarray(cluster_counts)
                    largest_cluster = np.argmax(cluster_counts)
                    largest_mask = cluster_ids == largest_cluster
                    removed = len(vert_tris) - largest_mask.sum()
                    if removed > 0:
                        print(f"  Vertebra {vi}: removed {removed} artefact triangles "
                              f"({cluster_counts.size - 1} small clusters)")
                    cleaned_triangles.append(vert_tris[largest_mask])
                filtered_triangles = np.concatenate(cleaned_triangles, axis=0)

                # Build KDTree from depth point cloud for distance-based intensity
                depth_points = capture.depth_point_cloud.reshape(
                    (-1, 3)).astype(np.float64)
                valid_mask = depth_points[:, 2] > 0
                depth_points = depth_points[valid_mask]
                surface_pcd = o3d.geometry.PointCloud()
                surface_pcd.points = o3d.utility.Vector3dVector(depth_points)
                kdtree = o3d.geometry.KDTreeFlann(surface_pcd)

                # Compute per-vertex distance to nearest point cloud surface point
                vertex_distances = np.zeros(len(vertices))
                for i, v in enumerate(vertices):
                    _, _, dist_sq = kdtree.search_knn_vector_3d(v, 1)
                    vertex_distances[i] = np.sqrt(dist_sq[0])

                # Map distance to intensity: 0mm -> 1.0, max_dist_mm -> 0.0
                max_dist_mm = 50.0  # vertices beyond this distance are invisible
                vertex_intensity = np.clip(
                    1.0 - vertex_distances / max_dist_mm, 0.0, 1.0)
                # Boost: raise floor so visible triangles stay vivid
                visible = vertex_intensity > 0.0
                vertex_intensity[visible] = 0.4 + \
                    0.6 * vertex_intensity[visible]
                print(f"Vertex distances to surface: mean={vertex_distances.mean():.1f}mm, "
                      f"median={np.median(vertex_distances):.1f}mm, max={vertex_distances.max():.1f}mm")

                # Transform kept vertices from depth camera space to color camera space
                vertices_color_space = (
                    T_depth_to_color[:3, :3] @ vertices.T).T + T_depth_to_color[:3, 3]

                # Project 3D vertices to 2D color image
                pts_2d, _ = cv2.projectPoints(
                    vertices_color_space,
                    np.zeros(3), np.zeros(3),
                    K_color, dist_color
                )
                pts_2d = pts_2d.reshape(-1, 2).astype(int)

                # Assign a distinct saturated color per vertebra (BGR)
                bright_colors_bgr = [
                    [0, 255, 255],   # yellow
                    [0, 255, 0],     # green
                    [255, 0, 255],   # magenta
                    [255, 255, 0],   # cyan
                    [0, 128, 255],   # orange
                    [255, 0, 0],     # blue
                    [0, 0, 255],     # red
                    [255, 255, 255],  # white
                ]
                vertebra_base_colors = {}
                for rank, vi in enumerate(keep_indices):
                    vertebra_base_colors[vi] = np.array(
                        bright_colors_bgr[rank % len(bright_colors_bgr)], dtype=np.float64)

                # Determine per-triangle vertebra label (majority vote of 3 vertices)
                tri_labels = vertex_labels[filtered_triangles]
                tri_vertebra = np.array([
                    np.bincount(row).argmax() for row in tri_labels
                ])

                # Draw filled triangles with distance-based intensity (dimmest first, brightest last)
                tri_intensities = vertex_intensity[filtered_triangles].mean(
                    axis=1)
                draw_order = np.argsort(tri_intensities)
                overlay = annotated.copy()
                for idx in draw_order:
                    tri = filtered_triangles[idx]
                    tri_intensity = tri_intensities[idx]
                    if tri_intensity < 0.01:
                        continue
                    base = vertebra_base_colors[tri_vertebra[idx]]
                    color = tuple(int(c * tri_intensity) for c in base)
                    pts_tri = np.array(
                        [pts_2d[tri[0]], pts_2d[tri[1]], pts_2d[tri[2]]])
                    cv2.fillPoly(overlay, [pts_tri], color)
                # Blend overlay with original
                alpha = 0.7
                annotated = cv2.addWeighted(
                    overlay, alpha, annotated, 1.0 - alpha, 0)

                print(
                    f"Overlaid mesh ({len(filtered_triangles)}/{len(triangles)} triangles) from: {overlay_mesh_path}")
            else:
                print(
                    f"Warning: Mesh has no vertices or vertex colors: {overlay_mesh_path}")

        # Display using matplotlib
        plt.figure(figsize=(12, 8))
        # Keep as BGR - matplotlib will interpret it as RGB, effectively swapping R and B
        plt.imshow(annotated)
        plt.title("ArUco Marker Detection - Single Frame")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Get depth point cloud for 3D visualization
        if capture.depth is not None:
            print("\nGenerating 3D visualization with point cloud and mesh overlay...")

            # Get point cloud from depth (native depth camera space)
            points = capture.depth_point_cloud.reshape(
                (-1, 3)).astype('float64')
            # Use transformed color for point cloud coloring (aligned to depth)
            colors_pc = transformed_color_bgr.reshape(
                (-1, 3)) if transformed_color_bgr is not None else np.full_like(points, 128)

            # Filter out zero/invalid points
            valid_mask = (points[:, 2] > 0)
            points = points[valid_mask]
            colors_pc = colors_pc[valid_mask]

            # Azure Kinect point cloud coordinate system:
            # X: right, Y: down, Z: forward (away from camera)
            # This matches OpenCV/ArUco convention, so no flip should be needed
            # However, if there's a mismatch, try uncommenting one of these:
            # Option 1: Flip Y axis
            # points[:, 1] = -points[:, 1]
            # Option 2: Flip X and Y axes
            # points[:, 0] = -points[:, 0]
            # points[:, 1] = -points[:, 1]

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(
                (colors_pc / 255).astype('float64'))
            print(f"Point cloud has {len(pcd.points)} points")

            # Prepare geometries for visualization
            geometries = [pcd]

            # Add camera coordinate frame at origin
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=100.0, origin=[0, 0, 0]
            )
            geometries.append(camera_frame)

            # Load and transform marker mesh for each detected tool
            for name in tracker.GetTrackableNames():
                found, T_marker_to_color, rvec, tvec, metrics = tracker.RequestPose(
                    name)
                if found:
                    # Chain transforms to get depth camera space
                    T_marker_to_depth = T_color_to_depth @ T_marker_to_color.astype(
                        np.float64)

                    print(f"\nLoading marker mesh: {tracker.marker_stl}")
                    marker_mesh = o3d.io.read_triangle_mesh(tracker.marker_stl)

                    if marker_mesh.has_vertices():
                        marker_mesh.compute_vertex_normals()

                        # Debug: Print transform details
                        print(
                            f"\n  Transform matrix for {name} (depth camera space):")
                        print(
                            f"  Translation (mm): [{T_marker_to_depth[0,3]:.1f}, {T_marker_to_depth[1,3]:.1f}, {T_marker_to_depth[2,3]:.1f}]")

                        # Get mesh center before transform
                        mesh_center_before = marker_mesh.get_center()
                        print(
                            f"  Mesh center before transform: {mesh_center_before}")

                        # Apply the transform to place mesh in depth camera coordinates
                        marker_mesh.transform(T_marker_to_depth)

                        # Get mesh center after transform
                        mesh_center_after = marker_mesh.get_center()
                        print(
                            f"  Mesh center after transform: {mesh_center_after}")

                        # Find nearest point cloud points to mesh center for comparison
                        distances = np.linalg.norm(np.asarray(
                            pcd.points) - mesh_center_after, axis=1)
                        nearest_idx = np.argmin(distances)
                        nearest_dist = distances[nearest_idx]
                        print(
                            f"  Distance from mesh center to nearest point cloud point: {nearest_dist:.1f} mm")

                        # Color the mesh for visibility
                        marker_mesh.paint_uniform_color(
                            [0.0, 1.0, 0.0])  # Green

                        geometries.append(marker_mesh)

                        # Add coordinate frame at the marker origin (transformed to depth space)
                        marker_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=50.0, origin=[0, 0, 0]
                        )
                        marker_frame.transform(T_marker_to_depth)
                        geometries.append(marker_frame)

                        print(
                            f"  Mesh transformed to depth camera space for tool: {name}")
                    else:
                        print(
                            f"  Warning: Could not load mesh from {tracker.marker_stl}")

            # Visualize
            print("\n" + "="*60)
            print("3D Visualization")
            print("="*60)
            print("Green mesh: Marker STL transformed to depth camera space")
            print("Colored points: Azure Kinect point cloud")
            print("Large RGB axes (100mm): Camera coordinate frame")
            print("Medium RGB axes (50mm): Marker coordinate frame")
            print("\nControls:")
            print("  - Mouse: Rotate view")
            print("  - Scroll: Zoom in/out")
            print("  - Ctrl + Mouse: Pan")
            print("  - Q or ESC: Close window")

            o3d.visualization.draw_geometries(
                geometries,
                window_name="ArUco Tracking - Point Cloud with Mesh Overlay",
                width=1280,
                height=720
            )

    playback.close()


if __name__ == "__main__":
    main()
