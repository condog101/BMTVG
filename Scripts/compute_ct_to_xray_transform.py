#!/usr/bin/env python3
"""
Compute CT to X-ray Source Transform

This script computes the transformation that positions the CT volume
in the original C-arm coordinate system (Camera 1's frame, with the
X-ray source at the origin).

Given:
- CT_to_World: CT's pose in ImFusion's world coordinate system (from ImFusion)
- Camera1_aligned_pose: The pose of Camera 1 after alignment to CT
  (Camera 1's original pose is identity, so original frame = world origin)

The aligned pose tells us where Camera 1 moved to for registration.
To transform points from CT/ImFusion world back to Camera 1's original frame,
we need the inverse of the aligned pose.

Output:
- CT_to_XraySource: The CT's pose in Camera 1's original coordinate system

The composition is:
    T_align_inv = inv(Camera1_aligned_pose)
    CT_to_XraySource = T_align_inv @ CT_to_World

This allows you to position the CT volume relative to the X-ray source.

Author: CT to X-ray Transform Script
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
        description='Compute CT to X-ray source transformation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script computes the transformation to position the CT volume
in the original C-arm coordinate system (X-ray source at origin).

Camera 1's original pose is identity (defines world origin).
After registration, Camera 1 moves to a new pose (camera1_aligned_pose).
We invert this to transform CT points back to the original X-ray frame.

Example usage:
  python compute_ct_to_xray_transform.py --ct-to-world ct_to_world.npy
  python compute_ct_to_xray_transform.py --ct-to-world ct_to_world.txt --camera1-aligned-pose alignment_results/camera1_aligned_pose.npy
        """
    )

    parser.add_argument(
        '--ct-to-world',
        type=str,
        default=None,
        help='Path to CT_to_World transform file (.npy or .txt). If not provided, uses built-in ImFusion values.'
    )
    parser.add_argument(
        '--camera1-aligned-pose',
        type=str,
        default='alignment_results/camera1_aligned_pose.npy',
        help='Path to Camera 1 aligned pose file (default: alignment_results/camera1_aligned_pose.npy). This is the pose Camera 1 moves to after registration.'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='alignment_results',
        help='Output directory (default: alignment_results)'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print(" CT TO X-RAY SOURCE TRANSFORM COMPUTATION")
    print("="*80)

    # Load or define CT_to_World
    if args.ct_to_world:
        print(f"\nLoading CT_to_World from: {args.ct_to_world}")
        CT_to_World = load_transform(args.ct_to_world)
    else:
        print("\nUsing built-in ImFusion CT_to_World transform")
        # From user's ImFusion export
        CT_to_World = np.array([
            [-1, -1.22464679914735e-16, 0, -25.2000003755093],
            [-1.22464679914735e-16, 1, 1.22464679914735e-16, 25.2000003755093],
            [-1.49975978266186e-32, 1.22464679914735e-16, -1, 38.6000005751848],
            [0, 0, 0, 1]
        ])

    print_transform(CT_to_World, "CT_to_World (CT voxels → ImFusion world)")

    # Load Camera 1 aligned pose (the pose Camera 1 moves to after registration)
    # Camera 1's original pose is identity (defines the world origin / X-ray source frame)
    # To transform from ImFusion world back to original Camera 1 frame, we need the inverse
    print(f"\nLoading Camera1_aligned_pose from: {args.camera1_aligned_pose}")
    if Path(args.camera1_aligned_pose).exists():
        Camera1_aligned_pose = load_transform(args.camera1_aligned_pose)
    else:
        print("  File not found, using built-in Camera 1 aligned pose")
        # This is the pose Camera 1 moves to after alignment/registration with CT
        Camera1_aligned_pose = np.array([
            [-0.642219834803108, -0.308811815184398,
                0.701561790997761,  -49.7773961909768],
            [-0.699996201194378,  0.609230576493043, -
                0.372616992338508,  -42.4210918939009],
            [-0.312344364602452, -0.730392611866024, -
                0.607427057705383,  199.835180029416],
            [0,                  0,                  0,                  1]
        ])
    print_transform(
        Camera1_aligned_pose, "Camera1_aligned_pose (Camera 1's pose after registration)")

    # Compute T_align_inv: transforms from aligned/ImFusion world to original Camera 1 frame
    # Since Camera 1 originally was at identity, and moved to Camera1_aligned_pose,
    # to go back to the original frame we need the inverse
    T_align_inv = np.linalg.inv(Camera1_aligned_pose)
    print_transform(
        T_align_inv, "T_align_inv = inv(Camera1_aligned_pose) (ImFusion world → original Camera 1 frame)")

    # Compute the composition
    # CT_to_XraySource = T_align_inv @ CT_to_World
    # This takes: CT voxels → ImFusion world → Camera 1 world
    CT_to_XraySource = T_align_inv @ CT_to_World

    print("\n" + "="*80)
    print("RESULT: CT TO X-RAY SOURCE TRANSFORM")
    print("="*80)
    print_transform(CT_to_XraySource,
                    "CT_to_XraySource (CT voxels → X-ray source frame)")

    # Also compute inverse (useful for projecting X-ray source points into CT)
    XraySource_to_CT = np.linalg.inv(CT_to_XraySource)
    print_transform(XraySource_to_CT,
                    "XraySource_to_CT (X-ray source frame → CT voxels)")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "="*80)
    print(f"SAVING RESULTS to {output_dir}/")
    print("="*80)

    save_transform(
        CT_to_XraySource,
        output_dir / 'ct_to_xray_source',
        'CT to X-ray source transform (4x4)\nPositions CT volume in Camera 1 coordinate system (X-ray source at origin)\np_xray = CT_to_XraySource @ p_ct'
    )

    save_transform(
        XraySource_to_CT,
        output_dir / 'xray_source_to_ct',
        'X-ray source to CT transform (4x4)\nTransforms points from Camera 1 frame to CT voxel coordinates\np_ct = XraySource_to_CT @ p_xray'
    )

    # Create summary
    with open(output_dir / 'ct_xray_transform_summary.txt', 'w') as f:
        f.write("CT to X-ray Source Transform Summary\n")
        f.write("="*80 + "\n\n")

        f.write(
            "This transform positions the CT volume in Camera 1's coordinate system,\n")
        f.write("where the X-ray source is at the origin.\n\n")

        f.write("Transform chain:\n")
        f.write(
            "  CT voxels → ImFusion world → Camera 1 world (X-ray source frame)\n\n")

        f.write("Usage:\n")
        f.write("  p_xray = CT_to_XraySource @ p_ct  (CT voxels to X-ray frame)\n")
        f.write("  p_ct = XraySource_to_CT @ p_xray  (X-ray frame to CT voxels)\n\n")

        f.write("="*80 + "\n")
        f.write("CT_to_XraySource (4x4):\n")
        f.write("="*80 + "\n")
        for row in CT_to_XraySource:
            f.write(
                f"  [{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f} {row[3]:12.6f}]\n")

        t = CT_to_XraySource[:3, 3]
        f.write(f"\nTranslation: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] mm\n")

        f.write("\n" + "="*80 + "\n")
        f.write("XraySource_to_CT (4x4):\n")
        f.write("="*80 + "\n")
        for row in XraySource_to_CT:
            f.write(
                f"  [{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f} {row[3]:12.6f}]\n")

        t_inv = XraySource_to_CT[:3, 3]
        f.write(
            f"\nTranslation: [{t_inv[0]:.2f}, {t_inv[1]:.2f}, {t_inv[2]:.2f}] mm\n")

    print(f"  ✓ ct_xray_transform_summary.txt")

    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)
    print(
        f"\nThe CT to X-ray source transform has been saved to {output_dir}/")
    print("Use CT_to_XraySource to position the CT relative to the X-ray source.")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
