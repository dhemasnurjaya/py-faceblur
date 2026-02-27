"""Output generation module."""

import colorsys
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

from .video import Frame
from .detect import FaceData
from .cluster import Cluster


def generate_output(
    frames: List[Frame], faces: List[FaceData], clusters: List[Cluster], output_dir: str
) -> None:
    """Generate output with bounding boxes and face crops.

    Args:
        frames: List of extracted frames
        faces: All detected faces
        clusters: Clustered faces
        output_dir: Output directory
    """
    output_dir = Path(output_dir)

    face_to_cluster: Dict[int, int] = {}
    for cluster in clusters:
        for face in cluster.faces:
            face_to_cluster[face.id] = cluster.id

    faces_by_frame: Dict[int, List[FaceData]] = {}
    for face in faces:
        if face.frame_index not in faces_by_frame:
            faces_by_frame[face.frame_index] = []
        faces_by_frame[face.frame_index].append(face)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        frame_faces = faces_by_frame.get(frame.index, [])
        _draw_frame_with_boxes(frame.path, frame_faces, face_to_cluster, frames_dir)

    for i, cluster in enumerate(clusters):
        if cluster.id == -1:
            cluster_dir = output_dir / "unclustered"
        else:
            cluster_dir = output_dir / f"cluster_{i:02d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        for face in cluster.faces:
            _extract_face_crop(face, cluster_dir)


def _get_cluster_color(cluster_id: int) -> Tuple[int, int, int]:
    """Generate a consistent color for a cluster."""
    if cluster_id < 0:
        return (128, 128, 128)

    hue = (cluster_id * 0.618033988749895) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


def _draw_frame_with_boxes(
    frame_path: Path,
    faces: List[FaceData],
    face_to_cluster: Dict[int, int],
    output_dir: Path,
) -> None:
    """Draw bounding boxes on a frame."""
    image = cv2.imread(str(frame_path))
    if image is None:
        return

    for face in faces:
        x1, y1, x2, y2 = face.bbox
        cluster_id = face_to_cluster.get(face.id, -1)
        color = _get_cluster_color(cluster_id)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if cluster_id >= 0:
            label = f"C{cluster_id}"
            cv2.putText(
                image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

    output_path = output_dir / f"frame_{frame_path.stem.split('_')[1]}.jpg"
    cv2.imwrite(str(output_path), image)


def _extract_face_crop(face: FaceData, output_dir: Path) -> None:
    """Extract and save a face crop."""
    image = cv2.imread(str(face.frame_path))
    if image is None:
        return

    x1, y1, x2, y2 = face.bbox
    face_img = image[y1:y2, x1:x2]

    if face_img.size == 0:
        return

    filename = f"face_{face.frame_index:04d}_{face.id % 100:02d}.jpg"
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), face_img)
