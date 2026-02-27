"""CLI module for faceblur-poc."""

import argparse
import sys
from pathlib import Path

from .video import extract_frames
from .detect import FaceDetector
from .cluster import cluster_faces
from .output import generate_output


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Face detection and clustering POC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    detect_parser = subparsers.add_parser(
        "detect", help="Detect and cluster faces in video"
    )
    detect_parser.add_argument(
        "--video", required=True, help="Path to input video file"
    )
    detect_parser.add_argument("--output", default="output", help="Output directory")
    detect_parser.add_argument(
        "--interval", type=int, default=30, help="Frame interval"
    )
    detect_parser.add_argument(
        "--eps", type=float, default=0.4, help="DBSCAN eps parameter (cosine distance)"
    )
    detect_parser.add_argument(
        "--min-samples", type=int, default=2, help="DBSCAN min_samples"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "detect":
        run_detect(args)


def run_detect(args):
    """Run the detect command."""
    print(f"Processing video: {args.video}")
    print(f"Output directory: {args.output}")
    print(f"Frame interval: {args.interval}")

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    print("\n[1/5] Extracting frames...")
    frames = extract_frames(args.video, f"{args.output}/frames_original", args.interval)
    print(f"Extracted {len(frames)} frames")

    if not frames:
        print("Error: No frames extracted from video", file=sys.stderr)
        sys.exit(1)

    print("\n[2/5] Initializing face detector...")
    detector = FaceDetector()

    print("\n[3/5] Detecting faces...")
    all_faces = []
    for i, frame in enumerate(frames):
        try:
            faces = detector.detect_faces(frame.path, frame.index)
            all_faces.extend(faces)
            print(
                f"  Frame {i + 1}/{len(frames)}: {len(faces)} faces (total: {len(all_faces)})"
            )
        except Exception as e:
            print(f"  Warning: Failed to detect faces in frame {frame.index}: {e}")

    print(f"Total faces detected: {len(all_faces)}")

    if not all_faces:
        print("Error: No faces detected in video", file=sys.stderr)
        sys.exit(1)

    print("\n[4/5] Clustering faces...")
    clusters = cluster_faces(all_faces, eps=args.eps, min_samples=args.min_samples)
    print(f"Found {len(clusters)} clusters")

    print("\n[5/5] Generating output...")
    generate_output(frames, all_faces, clusters, args.output)

    print(f"\nDone! Output saved to: {args.output}")
    print("  - frames/          : Frames with bounding boxes")
    for i, cluster in enumerate(clusters):
        if cluster.id == -1:
            print("  - unclustered/     : Single-occurrence faces")
        else:
            print(f"  - cluster_{i:02d}/     : Face crops for cluster {i}")

    detector.close()


if __name__ == "__main__":
    main()
