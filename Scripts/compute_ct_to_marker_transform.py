#!/usr/bin/env python3
"""
Compute CT to Marker Board Transform

This script computes the transformation that positions the CT volume
in the marker board coordinate system.

Given:
- CT_to_XraySource: CT's pose in Camera 1's frame (from compute_ct_to_xray_transform.py)
- T_xray_to_board: X-ray source to marker board transform (from detect_markers_and_compute_transform.py)

Output:
- CT_to_Marker: The CT's pose in marker board coordinates

The composition is:
    CT_to_Marker = T_xray_to_board @ CT_to_XraySource

This allows you to position the CT volume relative to the marker board,
which is useful for registration and overlay applications.

Author: CT to Marker Board Transform Script
Date: 2024-11-26
"""

import numpy as np
import argparse
from pathlib import Path


def load_transform(filepath: str) -> np.ndarray:
    """
    Load a 4x4 transform matrix from .npy or .txt file.

    Args:
        filepath: Path to transform file

    Returns:
        T: 4x4 transformation matrix
    """
    path = Path(filepath)

    if path.suffix == '.npy':
        T = np.load(filepath)
    elif path.suffix == '.txt':
        T = np.loadtxt(filepath)
    else:
        raise ValueError(f"Unknown file type: {path.suffix}")

    if T.shape != (4, 4):
        raise ValueError(f"Invalid transform shape: {T.shape}, expected (4,4)")

    return T


def print_transform(T: np.ndarray, name: str) -> None:
    """Print a transform matrix nicely."""
    print(f"\n{name}:")
    for row in T:
        print(f"  [{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f} {row[3]:12.6f}]")

    # Extract translation
    t = T[:3, 3]
    print(f"  Translation: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] mm")

    # Compute rotation angle
    R = T[:3, :3]
    trace = np.trace(R)
    # Clamp to valid range for arccos
    cos_angle = np.clip((trace - 1) / 2, -1, 1)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    print(f"  Rotation angle: {angle_deg:.2f}°")


def save_transform(T: np.ndarray, output_path: Path, description: str) -> None:
    """Save transform to both .npy and .txt formats."""
    # Save as .npy
    np.save(output_path.with_suffix('.npy'), T)

    # Save as .txt
    np.savetxt(
        output_path.with_suffix('.txt'),
        T,
        fmt='%.8f',
        header=description
    )

    print(f"  ✓ {output_path.with_suffix('.npy').name}")
    print(f"  ✓ {output_path.with_suffix('.txt').name}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute CT to marker board transformation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script computes the transformation to position the CT volume
in the marker board coordinate system.

Transform chain:
  CT voxels → X-ray source frame → Marker board coordinates

Example usage:
  python compute_ct_to_marker_transform.py detection_results/7501_transform.npy
  python compute_ct_to_marker_transform.py detection_results/7501_transform.npy --ct-to-xray alignment_results/ct_to_xray_source.npy
  python compute_ct_to_marker_transform.py detection_results/7501_transform.npy -o my_output_dir/
        """
    )

    parser.add_argument(
        'xray_to_board',
        type=str,
        help='Path to X-ray source to marker board transform file (.npy or .txt) from detect_markers_and_compute_transform.py'
    )
    parser.add_argument(
        '--ct-to-xray',
        type=str,
        default='alignment_results/ct_to_xray_source.npy',
        help='Path to CT_to_XraySource transform file (default: alignment_results/ct_to_xray_source.npy)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='alignment_results',
        help='Output directory (default: alignment_results)'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print(" CT TO MARKER BOARD TRANSFORM COMPUTATION")
    print("="*80)

    # Load CT_to_XraySource
    print(f"\nLoading CT_to_XraySource from: {args.ct_to_xray}")
    CT_to_XraySource = load_transform(args.ct_to_xray)
    print_transform(CT_to_XraySource,
                    "CT_to_XraySource (CT voxels → X-ray source frame)")

    # Load T_xray_to_board
    print(f"\nLoading T_xray_to_board from: {args.xray_to_board}")
    T_xray_to_board = load_transform(args.xray_to_board)
    print_transform(T_xray_to_board,
                    "T_xray_to_board (X-ray source → Marker board)")

    # Compute the composition
    # CT_to_Marker = T_xray_to_board @ CT_to_XraySource
    # This takes: CT voxels → X-ray source → Marker board
    CT_to_Marker = T_xray_to_board @ CT_to_XraySource

    print("\n" + "="*80)
    print("RESULT: CT TO MARKER BOARD TRANSFORM")
    print("="*80)
    print_transform(
        CT_to_Marker, "CT_to_Marker (CT voxels → Marker board coordinates)")

    # Also compute inverse (useful for transforming marker board points into CT)
    Marker_to_CT = np.linalg.inv(CT_to_Marker)
    print_transform(
        Marker_to_CT, "Marker_to_CT (Marker board coordinates → CT voxels)")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract image name from xray_to_board path for naming
    xray_to_board_path = Path(args.xray_to_board)
    image_name = xray_to_board_path.stem.replace('_transform', '')

    print(f"\n" + "="*80)
    print(f"SAVING RESULTS to {output_dir}/")
    print("="*80)

    save_transform(
        CT_to_Marker,
        output_dir / f'ct_to_marker_{image_name}',
        f'CT to marker board transform (4x4)\nPositions CT volume in marker board coordinate system\nSource image: {image_name}\np_marker = CT_to_Marker @ p_ct'
    )

    save_transform(
        Marker_to_CT,
        output_dir / f'marker_to_ct_{image_name}',
        f'Marker board to CT transform (4x4)\nTransforms points from marker board coordinates to CT voxel coordinates\nSource image: {image_name}\np_ct = Marker_to_CT @ p_marker'
    )

    # Create summary
    summary_path = output_dir / f'ct_marker_transform_summary_{image_name}.txt'
    with open(summary_path, 'w') as f:
        f.write("CT to Marker Board Transform Summary\n")
        f.write("="*80 + "\n\n")

        f.write(f"Source X-ray image: {image_name}\n\n")

        f.write("This transform positions the CT volume in marker board coordinates.\n")
        f.write("Useful for registration and overlay applications.\n\n")

        f.write("Transform chain:\n")
        f.write("  CT voxels → X-ray source frame → Marker board coordinates\n\n")

        f.write("Input transforms:\n")
        f.write(f"  CT_to_XraySource: {args.ct_to_xray}\n")
        f.write(f"  T_xray_to_board:  {args.xray_to_board}\n\n")

        f.write("Usage:\n")
        f.write("  p_marker = CT_to_Marker @ p_ct  (CT voxels to marker board)\n")
        f.write("  p_ct = Marker_to_CT @ p_marker  (marker board to CT voxels)\n\n")

        f.write("="*80 + "\n")
        f.write("CT_to_Marker (4x4):\n")
        f.write("="*80 + "\n")
        for row in CT_to_Marker:
            f.write(
                f"  [{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f} {row[3]:12.6f}]\n")

        t = CT_to_Marker[:3, 3]
        f.write(f"\nTranslation: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] mm\n")

        R = CT_to_Marker[:3, :3]
        trace = np.trace(R)
        cos_angle = np.clip((trace - 1) / 2, -1, 1)
        angle_deg = np.degrees(np.arccos(cos_angle))
        f.write(f"Rotation angle: {angle_deg:.2f}°\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Marker_to_CT (4x4):\n")
        f.write("="*80 + "\n")
        for row in Marker_to_CT:
            f.write(
                f"  [{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f} {row[3]:12.6f}]\n")

        t_inv = Marker_to_CT[:3, 3]
        f.write(
            f"\nTranslation: [{t_inv[0]:.2f}, {t_inv[1]:.2f}, {t_inv[2]:.2f}] mm\n")

    print(f"  ✓ {summary_path.name}")

    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)
    print(
        f"\nThe CT to marker board transform has been saved to {output_dir}/")
    print("Use CT_to_Marker to position the CT relative to the marker board.")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
