#!/usr/bin/env python3
"""
C-arm Calibration from ArUco Phantom and DICOM Images

This script performs complete C-arm calibration using:
- Multi-layer ArUco marker phantom (JSON geometry)
- DICOM X-ray images
- Standard camera calibration (suitable for ±15° range)

No prior knowledge of C-arm geometry required - everything determined from phantom.

Author: Calibration Script
Date: 2024
"""

import json
import numpy as np
import cv2
import cv2.aruco as aruco
import pydicom
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import sys


@dataclass
class CalibrationResult:
    """Store calibration results."""
    K: np.ndarray  # Intrinsic matrix (3x3)
    dist: np.ndarray  # Distortion coefficients
    rvecs: List[np.ndarray]  # Rotation vectors for each image
    tvecs: List[np.ndarray]  # Translation vectors for each image
    projection_matrices: List[np.ndarray]  # P = K[R|t] for each image
    reprojection_errors: List[float]  # Per-image reprojection errors
    image_filenames: List[str]  # Corresponding DICOM filenames
    rms_error: float  # Overall RMS reprojection error
    num_images: int  # Number of images used
    markers_per_image: List[int]  # Number of markers detected per image


class PhantomLoader:
    """Load and parse phantom geometry from JSON."""

    def __init__(self, json_path: str):
        """Load phantom configuration from JSON file."""
        print(f"\nLoading phantom from: {json_path}")
        with open(json_path, 'r') as f:
            self.config = json.load(f)

        self.marker_corners_3d = {}  # marker_id -> 4x3 array of corners
        self.layers = {}  # layer_name -> layer info
        self._parse_phantom()

    def _parse_phantom(self):
        """Parse phantom configuration and build marker dictionary."""
        print("\nPhantom Configuration:")
        print("-" * 60)

        for tool in self.config['toolList']:
            layer_id = tool['id']
            marker_ids = tool['marker_ids']
            marker_corners = tool['marker_corners_m']
            marker_count = tool['marker_count']

            # Extract Y-coordinate (depth) for this layer
            if marker_corners:
                # Y from first corner of first marker
                y_coord = marker_corners[0][0][1]
                depth_from_detector = abs(y_coord)
            else:
                depth_from_detector = 0

            print(f"\nLayer: {layer_id}")
            print(f"  Distance from detector: {depth_from_detector:.1f} mm")
            print(f"  Marker IDs: {marker_ids}")
            print(f"  Marker count: {marker_count}")

            # Store layer info
            self.layers[layer_id] = {
                'marker_ids': marker_ids,
                'marker_count': marker_count,
                'depth': depth_from_detector
            }

            # Parse each marker
            for marker_id, corners in zip(marker_ids, marker_corners):
                # Convert to numpy array (4 corners × 3 coordinates)
                corners_array = np.array(corners, dtype=np.float32)

                # Verify it's 4x3
                if corners_array.shape != (4, 3):
                    raise ValueError(
                        f"Marker {marker_id} has invalid corner shape: "
                        f"{corners_array.shape}, expected (4, 3)"
                    )

                # Store with marker ID as key
                self.marker_corners_3d[marker_id] = corners_array

                print(
                    f"    Marker {marker_id}: {corners_array.shape[0]} corners")

        print(f"\n{'='*60}")
        print(f"Total markers loaded: {len(self.marker_corners_3d)}")
        print(f"Total layers: {len(self.layers)}")
        print(f"{'='*60}")

    def apply_coordinate_transform(self, invert_y: bool = False, y_to_z: bool = False):
        """
        Apply coordinate system transformations to phantom geometry.

        Args:
            invert_y: If True, negate Y coordinates (higher Y becomes closer)
            y_to_z: If True, swap Y and Z axes (Y becomes depth)
        """
        if not invert_y and not y_to_z:
            return

        print("\nApplying coordinate transformations:")
        if invert_y:
            print("  - Inverting Y axis (higher Y → closer to detector)")
        if y_to_z:
            print("  - Swapping Y ↔ Z axes")

        for marker_id, corners in self.marker_corners_3d.items():
            # Make a copy to modify
            new_corners = corners.copy()

            if invert_y:
                # Find max Y to invert around
                max_y = max(c[:, 1].max()
                            for c in self.marker_corners_3d.values())
                new_corners[:, 1] = max_y - new_corners[:, 1]

            if y_to_z:
                # Swap Y and Z columns
                new_corners[:, [1, 2]] = new_corners[:, [2, 1]]

            self.marker_corners_3d[marker_id] = new_corners

    def get_marker_corners(self, marker_id: int) -> Optional[np.ndarray]:
        """Get 3D corners for a specific marker."""
        return self.marker_corners_3d.get(marker_id)

    def get_all_marker_ids(self) -> List[int]:
        """Get list of all marker IDs in phantom."""
        return sorted(self.marker_corners_3d.keys())

    def get_layer_info(self) -> Dict:
        """Get information about each layer."""
        return self.layers

    def get_marker_layer(self, marker_id: int) -> Optional[str]:
        """Determine which layer a marker belongs to."""
        for layer_name, info in self.layers.items():
            if marker_id in info['marker_ids']:
                return layer_name
        return None


class DicomImageLoader:
    """Load and preprocess DICOM images."""

    def __init__(self, dicom_dir: str):
        """Initialize with directory containing DICOM files."""
        self.dicom_dir = Path(dicom_dir)

        # Find all DICOM files
        self.dicom_files = sorted(list(self.dicom_dir.glob('*.dcm')))

        if not self.dicom_files:
            # Try also looking for .DCM (uppercase)
            self.dicom_files = sorted(list(self.dicom_dir.glob('*.DCM')))

        if not self.dicom_files:
            raise ValueError(
                f"No DICOM files (.dcm or .DCM) found in {dicom_dir}"
            )

        print(f"\nFound {len(self.dicom_files)} DICOM files in {dicom_dir}")

    def load_image(self, dicom_path: Path) -> Tuple[np.ndarray, dict]:
        """
        Load DICOM image and extract metadata.

        Returns:
            image: 8-bit grayscale numpy array
            metadata: dict with relevant DICOM tags
        """
        # Read DICOM with force=True to handle non-standard files
        dcm = pydicom.dcmread(str(dicom_path), force=True)

        # Handle missing transfer syntax by accessing raw pixel data
        try:
            image = dcm.pixel_array.astype(np.float32)
        except (AttributeError, KeyError) as e:
            # If transfer syntax is missing, try to read raw pixel data
            if hasattr(dcm, 'PixelData'):
                # Get image dimensions
                rows = dcm.Rows
                cols = dcm.Columns
                bits_allocated = getattr(dcm, 'BitsAllocated', 16)

                # Determine dtype based on bits allocated
                if bits_allocated == 8:
                    dtype = np.uint8
                elif bits_allocated == 16:
                    dtype = np.uint16
                else:
                    dtype = np.uint16  # Default to 16-bit

                # Read raw pixel data
                pixel_bytes = dcm.PixelData
                image = np.frombuffer(pixel_bytes, dtype=dtype)

                # Reshape to image dimensions
                expected_size = rows * cols
                if len(image) >= expected_size:
                    image = image[:expected_size].reshape(rows, cols)
                else:
                    raise ValueError(
                        f"Insufficient pixel data: expected {expected_size}, got {len(image)}")

                image = image.astype(np.float32)
            else:
                raise ValueError(
                    f"Unable to extract pixel data from {dicom_path}")

        # Handle photometric interpretation
        photometric = getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')
        if photometric == 'MONOCHROME1':
            # MONOCHROME1: lower values are brighter (inverted)
            image = image.max() - image

        # Normalize to 8-bit for ArUco detection
        image = self._normalize_to_8bit(image)

        # Extract metadata (if available, but not required)
        metadata = {
            'filename': dicom_path.name,
            'rows': dcm.Rows,
            'cols': dcm.Columns,
            'photometric': photometric,
        }

        # Optional metadata (may not be present)
        if hasattr(dcm, 'PixelSpacing'):
            metadata['pixel_spacing'] = dcm.PixelSpacing
        if hasattr(dcm, 'SOPInstanceUID'):
            metadata['sop_instance_uid'] = dcm.SOPInstanceUID

        return image, metadata

    def _normalize_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 8-bit range [0, 255] for processing.
        Uses robust normalization to handle outliers.
        """
        # Remove negative values if present
        if image.min() < 0:
            image = image - image.min()

        # Robust normalization using percentiles to avoid outlier issues
        p_low = np.percentile(image, 1)
        p_high = np.percentile(image, 99)

        # Clip and normalize
        image = np.clip(image, p_low, p_high)
        image = (image - p_low) / (p_high - p_low) * 255.0

        return image.astype(np.uint8)

    def load_all_images(self) -> List[Tuple[np.ndarray, dict]]:
        """Load all DICOM images from directory."""
        images = []

        print("\nLoading DICOM images:")
        print("-" * 60)

        for dcm_path in self.dicom_files:
            try:
                image, metadata = self.load_image(dcm_path)
                images.append((image, metadata))
                print(
                    f"  ✓ {dcm_path.name}: {image.shape[1]}×{image.shape[0]} pixels")
            except Exception as e:
                print(f"  ✗ {dcm_path.name}: Error - {e}")

        print(
            f"\nSuccessfully loaded: {len(images)}/{len(self.dicom_files)} images")

        return images


class ArucoDetector:
    """Detect ArUco markers in X-ray images."""

    def __init__(self, dict_type=aruco.DICT_4X4_50):
        """Initialize ArUco detector with specified dictionary."""
        self.aruco_dict = aruco.getPredefinedDictionary(dict_type)
        self.parameters = self._get_detection_parameters()

        # Create ArUco detector object (for OpenCV 4.7+)
        try:
            self.detector = aruco.ArucoDetector(
                self.aruco_dict, self.parameters)
            self.use_new_api = True
        except AttributeError:
            # Fall back to old API for older OpenCV versions
            self.detector = None
            self.use_new_api = False

        print(f"\nArUco Detector initialized:")
        print(f"  Dictionary: {dict_type}")
        print(
            f"  API version: {'New (4.7+)' if self.use_new_api else 'Legacy'}")

    def _get_detection_parameters(self) -> aruco.DetectorParameters:
        """
        Configure detection parameters optimized for X-ray images.

        X-ray images have:
        - Lower contrast than optical images
        - More noise
        - Possible scatter artifacts
        - Metal markers may have high contrast but noisy edges
        """
        parameters = aruco.DetectorParameters()

        # Adaptive thresholding for varying contrast
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.adaptiveThreshConstant = 7

        # Marker perimeter constraints (relaxed for X-ray)
        parameters.minMarkerPerimeterRate = 0.02  # Relaxed minimum
        parameters.maxMarkerPerimeterRate = 4.0   # Allow larger markers

        # Polygon approximation accuracy
        parameters.polygonalApproxAccuracyRate = 0.05

        # Corner detection parameters
        parameters.minCornerDistanceRate = 0.05
        parameters.minDistanceToBorder = 3
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 30
        parameters.cornerRefinementMinAccuracy = 0.1

        # Error correction (more lenient for noisy X-ray)
        parameters.maxErroneousBitsInBorderRate = 0.35
        parameters.errorCorrectionRate = 0.6

        # Marker separation
        parameters.minMarkerDistanceRate = 0.05

        return parameters

    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess X-ray image for better ArUco detection.

        Steps:
        1. CLAHE - enhance local contrast
        2. Denoise - reduce noise while preserving edges
        3. Optional sharpening
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Denoise using Non-local Means
        denoised = cv2.fastNlMeansDenoising(
            enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Optional: Light sharpening to enhance edges
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        return denoised

    def detect_markers(self, image: np.ndarray,
                       preprocess: bool = True) -> Tuple[List, np.ndarray, List]:
        """
        Detect ArUco markers in image.

        Args:
            image: Input grayscale image (8-bit)
            preprocess: Whether to apply preprocessing

        Returns:
            corners: List of detected corner arrays (each Nx4x2)
            ids: Array of marker IDs (Nx1), None if no markers detected
            rejected: List of rejected candidates
        """
        if preprocess:
            processed = self.preprocess_for_detection(image)
        else:
            processed = image

        # Detect markers using appropriate API
        if self.use_new_api:
            # New API (OpenCV 4.7+)
            corners, ids, rejected = self.detector.detectMarkers(processed)
        else:
            # Legacy API (OpenCV < 4.7)
            corners, ids, rejected = aruco.detectMarkers(
                processed,
                self.aruco_dict,
                parameters=self.parameters
            )

        return corners, ids, rejected

    def visualize_detection(self, image: np.ndarray, corners: List,
                            ids: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected markers on image.

        Args:
            image: Input image
            corners: Detected corners
            ids: Detected marker IDs
            save_path: Optional path to save visualization

        Returns:
            Visualization image (BGR color)
        """
        # Create color image for visualization
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()

        # Draw detected markers
        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(vis_image, corners, ids)

            # Add text with marker IDs
            for i, marker_id in enumerate(ids.flatten()):
                corner = corners[i][0]
                center = corner.mean(axis=0).astype(int)
                cv2.putText(
                    vis_image,
                    f"ID:{marker_id}",
                    (center[0] - 20, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, vis_image)

        return vis_image


class CarmCalibrator:
    """Calibrate C-arm using detected markers and known phantom geometry."""

    def __init__(self, phantom: PhantomLoader, detector: ArucoDetector,
                 intrinsics_json: dict = None, intrinsics_path: str = None):
        """Initialize calibrator with phantom and detector."""
        self.phantom = phantom
        self.detector = detector
        self._intrinsics_json = intrinsics_json
        self._intrinsics_path = intrinsics_path

        # Cache detection results to avoid re-detecting in two-pass calibration
        self._cached_detections = None

    def calibrate(self, images: List[Tuple[np.ndarray, dict]],
                  visualize: bool = True,
                  vis_output_dir: Optional[str] = None,
                  min_markers_per_image: int = 3,
                  min_layers_per_image: int = 2,
                  max_reproj_error: Optional[float] = None) -> CalibrationResult:
        """
        Perform full calibration from multiple images.

        If max_reproj_error is specified, performs two-pass calibration:
        1. Initial calibration with all images
        2. Re-calibration excluding high-error outliers

        Args:
            images: List of (image, metadata) tuples
            visualize: Whether to create visualization images
            vis_output_dir: Directory to save visualizations
            min_markers_per_image: Minimum markers required per image
            min_layers_per_image: Minimum layers required per image
            max_reproj_error: If set, exclude images with error > this value and re-calibrate

        Returns:
            CalibrationResult with calibration parameters
        """

        all_object_points = []
        all_image_points = []
        valid_images = []
        image_filenames = []
        markers_per_image = []

        if vis_output_dir:
            Path(vis_output_dir).mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("DETECTING MARKERS IN IMAGES")
        print("="*60)

        for idx, (image, metadata) in enumerate(images):
            print(f"\nImage {idx+1}/{len(images)}: {metadata['filename']}")

            # Detect markers
            corners_2d, detected_ids, rejected = self.detector.detect_markers(
                image)

            # Always visualize if requested, even with no detections
            if visualize and vis_output_dir:
                vis_path = Path(
                    vis_output_dir) / f"detection_{idx:03d}_{Path(metadata['filename']).stem}.png"
                self._create_debug_visualization(
                    image, corners_2d, detected_ids, rejected,
                    metadata['filename'], str(vis_path)
                )

            if detected_ids is None or len(detected_ids) == 0:
                print(f"  ✗ No markers detected, skipping")
                continue

            detected_ids_flat = detected_ids.flatten()
            print(
                f"  ✓ Detected {len(detected_ids_flat)} markers: {list(detected_ids_flat)}")

            # Validate detection quality
            is_valid, validation_msg = self._validate_detection(
                detected_ids_flat,
                min_markers=min_markers_per_image,
                min_layers=min_layers_per_image
            )

            if not is_valid:
                print(f"  ✗ {validation_msg}, skipping")
                continue

            print(f"  ✓ {validation_msg}")

            # Match detected markers with phantom geometry
            obj_points, img_points = self._match_markers_to_phantom(
                corners_2d, detected_ids_flat
            )

            min_corners = min_markers_per_image * 4
            if len(obj_points) < min_corners:
                print(
                    f"  ✗ Only {len(obj_points)} corners matched, need ≥{min_corners}")
                continue

            all_object_points.append(obj_points)
            all_image_points.append(img_points)
            valid_images.append(image)
            image_filenames.append(metadata['filename'])
            markers_per_image.append(len(detected_ids_flat))

            print(f"  ✓ Matched {len(obj_points)} corner points")

        if len(all_object_points) == 0:
            raise ValueError(
                "No valid images for calibration! "
                "Check that ArUco markers are visible and detectable."
            )

        print(f"\n{'='*60}")
        print(f"CALIBRATING WITH {len(all_object_points)} VALID IMAGES")
        print(f"{'='*60}")

        # Perform calibration
        image_size = (valid_images[0].shape[1], valid_images[0].shape[0])

        print(f"\nImage size: {image_size[0]}×{image_size[1]} pixels")
        print(f"Total 3D points: {sum(len(pts) for pts in all_object_points)}")
        print(f"Total 2D points: {sum(len(pts) for pts in all_image_points)}")

        # Validate phantom geometry
        print("\nValidating phantom geometry...")
        self._validate_phantom_geometry(all_object_points)

        print("\nRunning calibration...")

        # Initial intrinsic matrix for non-planar phantom
        if self._intrinsics_json is not None:
            K_initial = np.array(
                self._intrinsics_json['camera_matrix'], dtype=np.float64)
            f = K_initial[0, 0]
            cx = K_initial[0, 2]
            cy = K_initial[1, 2]
            print(
                f"\nInitial intrinsic matrix (from {self._intrinsics_path}):")
        else:
            # Based on C-arm geometry: f = 450.0mm / 0.195mm/pixel ≈ 2307.7 pixels
            f = 450.0 / 0.195  # ~2307.7 pixels
            cx = image_size[0] / 2.0  # Principal point x (image center)
            cy = image_size[1] / 2.0  # Principal point y (image center)

            K_initial = np.array([
                [f,    0,    cx],
                [0,    f,    cy],
                [0,    0,    1.0]
            ], dtype=np.float64)
            print(f"\nInitial intrinsic matrix (hardcoded estimate):")

        print(f"  Focal length: {f:.2f} pixels")
        print(f"  Principal point: ({cx:.1f}, {cy:.1f})")

        # Calibration flags for non-planar phantom
        # Fix principal point to reduce DOF and improve stability with noisy marker detections
        calib_flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS |  # Use K_initial as starting point
            cv2.CALIB_FIX_PRINCIPAL_POINT |  # Fix cx, cy at image center
            cv2.CALIB_FIX_ASPECT_RATIO |     # Fix fx = fy (square pixels)
            cv2.CALIB_ZERO_TANGENT_DIST |    # Assume no tangential distortion
            cv2.CALIB_FIX_K3                 # Only use k1, k2 distortion
        )

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            all_object_points,
            all_image_points,
            image_size,
            K_initial,  # Provide initial camera matrix
            None,       # Initial distortion coefficients (None = zeros)
            flags=calib_flags
        )

        print(f"\n✓ Calibration complete!")
        print(f"  RMS reprojection error: {ret:.4f} pixels")

        # Compute projection matrices and per-image errors
        projection_matrices = []
        reprojection_errors = []

        print("\nComputing projection matrices...")
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)

            # Build projection matrix P = K[R|t]
            RT = np.hstack([R, tvec])
            P = K @ RT
            projection_matrices.append(P)

            # Calculate per-image reprojection error
            img_points_reprojected, _ = cv2.projectPoints(
                all_object_points[i],
                rvec,
                tvec,
                K,
                dist
            )

            diff = all_image_points[i] - img_points_reprojected.reshape(-1, 2)
            error = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

            reprojection_errors.append(error)
            print(f"  Image {i+1}: {error:.4f} pixels")

        # Detect and optionally remove outlier images
        reprojection_errors_array = np.array(reprojection_errors)
        median_error = np.median(reprojection_errors_array)
        std_error = np.std(reprojection_errors_array)
        threshold = median_error + 2.0 * std_error

        outlier_indices = np.where(reprojection_errors_array > threshold)[0]

        if len(outlier_indices) > 0:
            print(
                f"\n⚠ Detected {len(outlier_indices)} outlier images with error > {threshold:.2f} pixels:")
            for idx in outlier_indices:
                print(
                    f"    Image {idx+1} ({image_filenames[idx]}): {reprojection_errors[idx]:.2f} px")
            print(f"\n  Median error (good images): {median_error:.2f} pixels")

            # Two-pass calibration: re-calibrate excluding outliers if requested
            if max_reproj_error is not None:
                # Filter out images with high reprojection errors
                good_indices = [i for i, err in enumerate(reprojection_errors)
                                if err <= max_reproj_error]

                if len(good_indices) < len(all_object_points) and len(good_indices) >= 4:
                    print(f"\n{'='*60}")
                    print(
                        f"PASS 2: RE-CALIBRATING WITH {len(good_indices)} GOOD IMAGES")
                    print(
                        f"(Excluding {len(all_object_points) - len(good_indices)} images with error > {max_reproj_error:.2f} px)")
                    print(f"{'='*60}")

                    # Filter the data
                    all_object_points_filtered = [
                        all_object_points[i] for i in good_indices]
                    all_image_points_filtered = [
                        all_image_points[i] for i in good_indices]
                    image_filenames_filtered = [
                        image_filenames[i] for i in good_indices]
                    markers_per_image_filtered = [
                        markers_per_image[i] for i in good_indices]

                    # Re-run calibration with same initial guess
                    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                        all_object_points_filtered,
                        all_image_points_filtered,
                        image_size,
                        K_initial,
                        None,
                        flags=calib_flags
                    )

                    print(f"\n✓ Pass 2 calibration complete!")
                    print(
                        f"  RMS reprojection error: {ret:.4f} pixels (was {reprojection_errors_array.mean():.4f})")

                    # Recompute projection matrices and errors
                    projection_matrices = []
                    reprojection_errors = []

                    print("\nRecomputing projection matrices...")
                    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                        R, _ = cv2.Rodrigues(rvec)
                        RT = np.hstack([R, tvec])
                        P = K @ RT
                        projection_matrices.append(P)

                        img_points_reprojected, _ = cv2.projectPoints(
                            all_object_points_filtered[i],
                            rvec, tvec, K, dist
                        )

                        diff = all_image_points_filtered[i] - img_points_reprojected.reshape(-1, 2)
                        error = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

                        reprojection_errors.append(error)
                        print(f"  Image {i+1}: {error:.4f} pixels")

                    # Update with filtered data
                    all_object_points = all_object_points_filtered
                    all_image_points = all_image_points_filtered
                    image_filenames = image_filenames_filtered
                    markers_per_image = markers_per_image_filtered
                    valid_images = [valid_images[i] for i in good_indices]

                elif len(good_indices) < 4:
                    print(
                        f"\n⚠ Only {len(good_indices)} good images - need at least 4. Keeping all images.")
            else:
                print(
                    f"  Tip: Use --max-reproj-error {max(5.0, median_error * 2):.1f} to auto-exclude outliers.")

        # Compute max angular disparity between views
        max_angle = 0.0
        for i in range(len(rvecs)):
            R_i, _ = cv2.Rodrigues(rvecs[i])
            for j in range(i + 1, len(rvecs)):
                R_j, _ = cv2.Rodrigues(rvecs[j])
                R_rel = R_i @ R_j.T
                angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
                if angle > max_angle:
                    max_angle = angle
        print(f"\n  Max angular disparity between views: {max_angle:.2f}°")

        return CalibrationResult(
            K=K,
            dist=dist,
            rvecs=rvecs,
            tvecs=tvecs,
            projection_matrices=projection_matrices,
            reprojection_errors=reprojection_errors,
            image_filenames=image_filenames,
            rms_error=ret,
            num_images=len(valid_images),
            markers_per_image=markers_per_image
        )

    def _validate_phantom_geometry(self, all_object_points: List[np.ndarray]):
        """
        Validate that phantom geometry makes sense.
        Check for coordinate system issues or unrealistic scales.
        """
        # Combine all 3D points
        all_pts = np.vstack(all_object_points)

        # Check ranges
        x_range = all_pts[:, 0].max() - all_pts[:, 0].min()
        y_range = all_pts[:, 1].max() - all_pts[:, 1].min()
        z_range = all_pts[:, 2].max() - all_pts[:, 2].min()

        print(f"  3D point cloud spans:")
        print(
            f"    X: {x_range:.2f} mm ({all_pts[:, 0].min():.2f} to {all_pts[:, 0].max():.2f})")
        print(
            f"    Y: {y_range:.2f} mm ({all_pts[:, 1].min():.2f} to {all_pts[:, 1].max():.2f})")
        print(
            f"    Z: {z_range:.2f} mm ({all_pts[:, 2].min():.2f} to {all_pts[:, 2].max():.2f})")

        # Warn if geometry seems unrealistic
        if x_range > 500 or y_range > 500 or z_range > 500:
            print(f"  ⚠ WARNING: Phantom spans > 500mm - check coordinate units!")

        if y_range < 10:
            print(f"  ⚠ WARNING: Y-range < 10mm - phantom may be too flat!")

    def _validate_detection(self, detected_ids: np.ndarray,
                            min_markers: int = 3,
                            min_layers: int = 2) -> Tuple[bool, str]:
        """
        Check if detected markers provide sufficient coverage.

        Args:
            detected_ids: Array of detected marker IDs
            min_markers: Minimum total markers required
            min_layers: Minimum layers with markers required

        Returns:
            (is_valid, message): Validation result and description
        """
        layer_info = self.phantom.get_layer_info()
        detected_set = set(detected_ids)

        layers_with_markers = 0
        total_markers = 0
        layer_details = []

        for layer_id, info in layer_info.items():
            layer_marker_ids = set(info['marker_ids'])
            detected_in_layer = len(detected_set & layer_marker_ids)

            if detected_in_layer > 0:
                layers_with_markers += 1
                total_markers += detected_in_layer
                layer_details.append(
                    f"{layer_id}:{detected_in_layer}/{info['marker_count']}"
                )

        # Build message
        msg = f"Layers: {layers_with_markers}/{len(layer_info)} ({', '.join(layer_details)})"

        # Check requirements
        if layers_with_markers < min_layers:
            return False, f"Only {layers_with_markers} layers (need ≥{min_layers})"

        if total_markers < min_markers:
            return False, f"Only {total_markers} markers (need ≥{min_markers})"

        return True, msg

    def _create_debug_visualization(self, image: np.ndarray, corners: List,
                                    ids: np.ndarray, rejected: List,
                                    filename: str, save_path: str):
        """
        Create comprehensive debug visualization showing detection process.

        Args:
            image: Original normalized image
            corners: Detected corners
            ids: Detected marker IDs
            rejected: Rejected candidates
            filename: Source filename
            save_path: Where to save visualization
        """
        # Preprocess image to show what detector sees
        preprocessed = self.detector.preprocess_for_detection(image)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Original image
        axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('Original (Normalized)',
                             fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Preprocessed image
        axes[0, 1].imshow(preprocessed, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title(
            'Preprocessed (CLAHE+Denoise+Sharpen)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # Detection result on preprocessed
        vis_image = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(vis_image, corners, ids)
            for i, marker_id in enumerate(ids.flatten()):
                corner = corners[i][0]
                center = corner.mean(axis=0).astype(int)
                cv2.putText(vis_image, f"ID:{marker_id}",
                            (center[0] - 30, center[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        axes[0, 2].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        title_color = 'green' if ids is not None and len(ids) > 0 else 'red'
        if ids is not None and len(ids) > 0:
            title_text = f'✓ Detected: {len(ids)} markers'
        else:
            title_text = '✗ No markers detected'
        axes[0, 2].set_title(
            title_text,
            fontsize=14, fontweight='bold', color=title_color
        )
        axes[0, 2].axis('off')

        # Histogram - Original
        axes[1, 0].hist(image.ravel(), bins=256, color='blue',
                        alpha=0.7, range=(0, 255))
        axes[1, 0].set_title('Original Histogram', fontsize=12)
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, 255)

        # Histogram - Preprocessed
        axes[1, 1].hist(preprocessed.ravel(), bins=256,
                        color='green', alpha=0.7, range=(0, 255))
        axes[1, 1].set_title('Preprocessed Histogram', fontsize=12)
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, 255)

        # Rejected candidates
        rejected_image = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        if rejected and len(rejected) > 0:
            for rejected_corners in rejected:
                cv2.polylines(rejected_image, [rejected_corners.astype(int)],
                              True, (0, 0, 255), 2)

        axes[1, 2].imshow(cv2.cvtColor(rejected_image, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(
            f'Rejected Candidates: {len(rejected) if rejected else 0}',
            fontsize=14, fontweight='bold'
        )
        axes[1, 2].axis('off')

        # Add info text
        info_text = f"File: {filename}\n"
        info_text += f"Image size: {image.shape[1]}×{image.shape[0]} pixels\n"
        info_text += f"Detected markers: {len(ids) if ids is not None else 0}\n"
        if ids is not None and len(ids) > 0:
            info_text += f"Marker IDs: {list(ids.flatten())}\n"
        info_text += f"Rejected candidates: {len(rejected) if rejected else 0}"

        fig.text(0.5, 0.02, info_text, ha='center', fontsize=11,
                 family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _match_markers_to_phantom(self, corners_2d: List,
                                  detected_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match detected 2D corners with known 3D phantom geometry.

        Args:
            corners_2d: List of detected corner arrays
            detected_ids: Array of detected marker IDs

        Returns:
            object_points: Nx3 array of 3D points
            image_points: Nx2 array of 2D points
        """
        obj_points = []
        img_points = []

        for i, marker_id in enumerate(detected_ids):
            # Get 3D corners from phantom
            corners_3d = self.phantom.get_marker_corners(marker_id)

            if corners_3d is None:
                print(
                    f"    ⚠ Marker {marker_id} not in phantom definition, skipping")
                continue

            # Get detected 2D corners (shape: 4x2)
            corners_2d_marker = corners_2d[i][0]

            # Verify shapes
            if corners_3d.shape[0] != 4 or corners_2d_marker.shape[0] != 4:
                print(
                    f"    ⚠ Marker {marker_id} has wrong number of corners, skipping")
                continue

            # Append all 4 corners
            obj_points.extend(corners_3d)
            img_points.extend(corners_2d_marker)

        return (
            np.array(obj_points, dtype=np.float32),
            np.array(img_points, dtype=np.float32)
        )


def save_calibration_results(result: CalibrationResult, output_dir: str):
    """
    Save calibration results to files.

    Creates:
    - intrinsic_matrix.txt: K matrix
    - distortion_coeffs.txt: Distortion coefficients
    - projection_matrix_XXX.txt: P matrix for each image
    - calibration_summary.txt: Human-readable summary
    - calibration_data.npz: All data in numpy format
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to {output_dir}/")

    # Save intrinsic matrix
    np.savetxt(
        output_path / 'intrinsic_matrix.txt',
        result.K,
        fmt='%.8f',
        header='Intrinsic matrix K (3x3)\nfx  0  cx\n 0 fy  cy\n 0  0   1'
    )

    # Save distortion coefficients
    np.savetxt(
        output_path / 'distortion_coeffs.txt',
        result.dist,
        fmt='%.8f',
        header='Distortion coefficients [k1, k2, p1, p2, k3, ...]'
    )

    # Save projection matrices
    for i, (P, filename) in enumerate(zip(result.projection_matrices,
                                          result.image_filenames)):
        np.savetxt(
            output_path /
            f'projection_matrix_{i:03d}_{Path(filename).stem}.txt',
            P,
            fmt='%.8f',
            header=f'Projection matrix P (3x4) for {filename}\nP = K[R|t]'
        )

    # Save rotation and translation vectors
    for i, (rvec, tvec, filename) in enumerate(zip(result.rvecs,
                                                   result.tvecs,
                                                   result.image_filenames)):
        R, _ = cv2.Rodrigues(rvec)

        with open(output_path / f'pose_{i:03d}_{Path(filename).stem}.txt', 'w') as f:
            f.write(f"Pose for {filename}\n")
            f.write("=" * 60 + "\n\n")
            f.write("Rotation vector (axis-angle):\n")
            f.write(f"{rvec.flatten()}\n\n")
            f.write("Rotation matrix R (3x3):\n")
            f.write(f"{R}\n\n")
            f.write("Translation vector t (3x1) [mm]:\n")
            f.write(f"{tvec.flatten()}\n")

    # Save comprehensive summary
    with open(output_path / 'calibration_summary.txt', 'w') as f:
        f.write("C-ARM CALIBRATION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("INTRINSIC PARAMETERS\n")
        f.write("-" * 60 + "\n")
        f.write("Intrinsic Matrix K:\n")
        f.write(f"{result.K}\n\n")

        fx, fy = result.K[0, 0], result.K[1, 1]
        cx, cy = result.K[0, 2], result.K[1, 2]

        f.write(f"Focal lengths:\n")
        f.write(f"  fx = {fx:.2f} pixels\n")
        f.write(f"  fy = {fy:.2f} pixels\n")
        f.write(f"  Aspect ratio: {fx/fy:.6f}\n\n")

        f.write(f"Principal point (optical center):\n")
        f.write(f"  cx = {cx:.2f} pixels\n")
        f.write(f"  cy = {cy:.2f} pixels\n\n")

        f.write(f"Distortion coefficients:\n")
        f.write(f"{result.dist.flatten()}\n\n")

        f.write("\nCALIBRATION QUALITY\n")
        f.write("-" * 60 + "\n")
        f.write(f"Number of calibrated views: {result.num_images}\n")
        f.write(
            f"Overall RMS reprojection error: {result.rms_error:.4f} pixels\n")
        f.write(
            f"Mean reprojection error: {np.mean(result.reprojection_errors):.4f} pixels\n")
        f.write(
            f"Std reprojection error: {np.std(result.reprojection_errors):.4f} pixels\n")
        f.write(
            f"Min reprojection error: {np.min(result.reprojection_errors):.4f} pixels\n")
        f.write(
            f"Max reprojection error: {np.max(result.reprojection_errors):.4f} pixels\n\n")

        f.write("\nPER-IMAGE DETAILS\n")
        f.write("-" * 60 + "\n")
        for i, (error, filename, num_markers) in enumerate(
            zip(result.reprojection_errors,
                result.image_filenames, result.markers_per_image)
        ):
            f.write(f"{i+1:3d}. {filename:40s} | "
                    f"{num_markers:2d} markers | "
                    f"{error:.4f} px\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Calibration completed successfully!\n")

    # Save all data in numpy format for easy loading
    np.savez(
        output_path / 'calibration_data.npz',
        K=result.K,
        dist=result.dist,
        rvecs=np.array(result.rvecs, dtype=object),
        tvecs=np.array(result.tvecs, dtype=object),
        projection_matrices=np.array(result.projection_matrices),
        reprojection_errors=np.array(result.reprojection_errors),
        rms_error=result.rms_error,
        image_filenames=np.array(result.image_filenames),
        markers_per_image=np.array(result.markers_per_image)
    )

    print("  ✓ intrinsic_matrix.txt")
    print("  ✓ distortion_coeffs.txt")
    print(f"  ✓ {len(result.projection_matrices)} projection matrices")
    print(f"  ✓ {len(result.projection_matrices)} pose files")
    print("  ✓ calibration_summary.txt")
    print("  ✓ calibration_data.npz")


def plot_calibration_quality(result: CalibrationResult, output_dir: str):
    """Create plots showing calibration quality."""
    output_path = Path(output_dir)

    print(f"\nGenerating quality plots...")

    # Plot 1: Reprojection errors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    indices = range(len(result.reprojection_errors))
    ax1.bar(indices, result.reprojection_errors, color='steelblue', alpha=0.7)
    ax1.axhline(
        np.mean(result.reprojection_errors),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {np.mean(result.reprojection_errors):.3f} px'
    )
    ax1.set_xlabel('Image Index', fontsize=12)
    ax1.set_ylabel('Reprojection Error (pixels)', fontsize=12)
    ax1.set_title('Per-Image Reprojection Errors',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Histogram
    ax2.hist(result.reprojection_errors, bins=15,
             color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(
        np.mean(result.reprojection_errors),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {np.mean(result.reprojection_errors):.3f} px'
    )
    ax2.set_xlabel('Reprojection Error (pixels)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'reprojection_errors.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Markers per image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(indices, result.markers_per_image, marker='o',
            linewidth=2, markersize=8, color='green')
    ax.axhline(
        np.mean(result.markers_per_image),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {np.mean(result.markers_per_image):.1f} markers'
    )
    ax.set_xlabel('Image Index', fontsize=12)
    ax.set_ylabel('Number of Detected Markers', fontsize=12)
    ax.set_title('Marker Detection Count per Image',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'markers_per_image.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print("  ✓ reprojection_errors.png")
    print("  ✓ markers_per_image.png")


def print_calibration_summary(result: CalibrationResult):
    """Print calibration summary to console."""
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)

    print("\nIntrinsic Matrix K:")
    print(result.K)

    fx, fy = result.K[0, 0], result.K[1, 1]
    cx, cy = result.K[0, 2], result.K[1, 2]

    print(f"\nFocal lengths:")
    print(f"  fx = {fx:.2f} pixels")
    print(f"  fy = {fy:.2f} pixels")
    print(f"  Aspect ratio: {fx/fy:.6f}")

    print(f"\nPrincipal point:")
    print(f"  cx = {cx:.2f} pixels")
    print(f"  cy = {cy:.2f} pixels")

    print(f"\nCalibration quality:")
    print(f"  Images used: {result.num_images}")
    print(f"  RMS error: {result.rms_error:.4f} pixels")
    print(f"  Mean error: {np.mean(result.reprojection_errors):.4f} pixels")
    print(f"  Std error: {np.std(result.reprojection_errors):.4f} pixels")
    print(f"  Error range: [{np.min(result.reprojection_errors):.4f}, "
          f"{np.max(result.reprojection_errors):.4f}] pixels")

    print(f"\n✓ Calibration complete!")


def main():
    """Main calibration pipeline."""
    parser = argparse.ArgumentParser(
        description='C-arm Calibration from ArUco Phantom and DICOM Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""


Example usage:
  python calibrate_carm.py phantom.json / path/to/dicoms/
  python calibrate_carm.py phantom.json / path/to/dicoms / --visualize - -output-dir results/
  python calibrate_carm.py phantom.json / path/to/dicoms / --min-markers 4 - -min-layers 3
        """
    )

    parser.add_argument(
        'phantom_json',
        type=str,
        help='Path to phantom geometry JSON file'
    )
    parser.add_argument(
        'dicom_dir',
        type=str,
        help='Directory containing DICOM (.dcm) images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='calibration_results_real',
        help='Output directory for results (default: calibration_results)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization images of marker detection'
    )
    parser.add_argument(
        '--aruco-dict',
        type=str,
        default='DICT_6X6_50',
        choices=['DICT_4X4_50', 'DICT_5X5_50', 'DICT_6X6_50', 'DICT_7X7_50',
                 'DICT_4X4_100', 'DICT_5X5_100', 'DICT_6X6_100', 'DICT_7X7_100'],
        help='ArUco dictionary type (default: DICT_6X6_50)'
    )
    parser.add_argument(
        '--min-markers',
        type=int,
        default=3,
        help='Minimum markers required per image (default: 3)'
    )
    parser.add_argument(
        '--min-layers',
        type=int,
        default=2,
        help='Minimum layers with markers required per image (default: 2)'
    )
    parser.add_argument(
        '--max-reproj-error',
        type=float,
        default=None,
        help='Maximum per-image reprojection error (pixels). Images above this are excluded.'
    )
    parser.add_argument(
        '--intrinsics',
        type=str,
        default=None,
        help='Path to calibration JSON with initial intrinsics (camera_matrix, dist_coeffs). '
             'If not provided, uses hardcoded SDD/pixel-size estimate.'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*60)
    print(" C-ARM CALIBRATION FROM ARUCO PHANTOM")
    print("="*60)
    print(f"\nPhantom: {args.phantom_json}")
    print(f"DICOM directory: {args.dicom_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"ArUco dictionary: {args.aruco_dict}")
    print(f"Minimum markers per image: {args.min_markers}")
    print(f"Minimum layers per image: {args.min_layers}")

    try:
        # Step 1: Load phantom geometry
        phantom = PhantomLoader(args.phantom_json)

        # Step 1b: Transform coordinate system for C-arm geometry
        # In C-arm: higher Y = closer to detector
        # For OpenCV: we need objects closer to camera to have smaller Z
        # So: invert Y axis (so closer objects have smaller values)
        phantom.apply_coordinate_transform(invert_y=True, y_to_z=False)

        # Step 2: Load DICOM images
        image_loader = DicomImageLoader(args.dicom_dir)
        images = image_loader.load_all_images()

        if len(images) == 0:
            print("\n✗ No images loaded successfully!")
            return 1

        # Step 3: Initialize detector
        dict_type = getattr(aruco, args.aruco_dict)
        detector = ArucoDetector(dict_type=dict_type)

        # Step 4: Perform calibration
        intrinsics_json = None
        if args.intrinsics:
            print(f"\nLoading initial intrinsics from: {args.intrinsics}")
            with open(args.intrinsics, 'r') as f:
                intrinsics_json = json.load(f)
            print(f"  fx={intrinsics_json['camera_matrix'][0][0]:.2f}, "
                  f"fy={intrinsics_json['camera_matrix'][1][1]:.2f}, "
                  f"cx={intrinsics_json['camera_matrix'][0][2]:.2f}, "
                  f"cy={intrinsics_json['camera_matrix'][1][2]:.2f}")

        calibrator = CarmCalibrator(phantom, detector,
                                    intrinsics_json=intrinsics_json,
                                    intrinsics_path=args.intrinsics)

        vis_dir = Path(args.output_dir) / \
            'visualizations' if args.visualize else None

        result = calibrator.calibrate(
            images,
            visualize=args.visualize,
            vis_output_dir=str(vis_dir) if vis_dir else None,
            min_markers_per_image=args.min_markers,
            min_layers_per_image=args.min_layers,
            max_reproj_error=args.max_reproj_error
        )

        # Step 5: Save results
        save_calibration_results(result, args.output_dir)
        plot_calibration_quality(result, args.output_dir)

        # Step 6: Print summary
        print_calibration_summary(result)

        print(f"\n{'='*60}")
        print(f"All results saved to: {args.output_dir}/")
        print(f"{'='*60}\n")

        return 0

    except FileNotFoundError as e:
        print(f"\n✗ File not found: {e}")
        return 1
    except ValueError as e:
        print(f"\n✗ Value error: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
