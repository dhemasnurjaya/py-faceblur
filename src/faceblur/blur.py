"""Face blur application logic."""

import cv2
import numpy as np
from typing import List, Tuple, Dict


BlurMethod = str  # "gaussian", "pixelate", "blackout", "elliptical", "median"


def apply_blur(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    method: BlurMethod = "gaussian",
    strength: float = 5.0,
) -> np.ndarray:
    """Apply blur to a face region in an image.

    Args:
        image: Input image (BGR, modified in-place)
        bbox: Face bounding box (x1, y1, x2, y2)
        method: Blur method name
        strength: Blur strength multiplier

    Returns:
        The modified image
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return image

    face_region = image[y1:y2, x1:x2]

    if method == "gaussian":
        ksize = int(face_region.shape[0] * strength) | 1  # Must be odd
        ksize = max(ksize, 3)
        blurred = cv2.GaussianBlur(face_region, (ksize, ksize), 0)
    elif method == "pixelate":
        ph = max(1, int(face_region.shape[0] / (strength * 2)))
        pw = max(1, int(face_region.shape[1] / (strength * 2)))
        small = cv2.resize(face_region, (pw, ph), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.resize(
            small,
            (face_region.shape[1], face_region.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    elif method == "blackout":
        blurred = np.zeros_like(face_region)
    elif method == "elliptical":
        ksize = int(face_region.shape[0] * strength) | 1
        ksize = max(ksize, 3)
        full_blur = cv2.GaussianBlur(face_region, (ksize, ksize), 0)
        mask = np.zeros(face_region.shape[:2], dtype=np.uint8)
        center = (face_region.shape[1] // 2, face_region.shape[0] // 2)
        axes = (face_region.shape[1] // 2, face_region.shape[0] // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask_3ch = cv2.merge([mask, mask, mask])
        blurred = np.where(mask_3ch > 0, full_blur, face_region)
    elif method == "median":
        ksize = int(face_region.shape[0] * strength) | 1
        ksize = max(ksize, 3)
        blurred = cv2.medianBlur(face_region, ksize)
    else:
        raise ValueError(f"Unknown blur method: {method}")

    image[y1:y2, x1:x2] = blurred
    return image


def interpolate_bboxes(
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
    t: float,
) -> Tuple[int, int, int, int]:
    """Linearly interpolate between two bounding boxes.

    Args:
        bbox_a: Bounding box at time 0
        bbox_b: Bounding box at time 1
        t: Interpolation factor (0.0 to 1.0)

    Returns:
        Interpolated bounding box
    """
    return tuple(int(a + t * (b - a)) for a, b in zip(bbox_a, bbox_b))


def get_bboxes_for_frame(
    frame_index: int,
    keyframe_bboxes: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]],
    keyframe_indices: List[int],
) -> List[Tuple[int, Tuple[int, int, int, int]]]:
    """Get bounding boxes for a frame by looking up or interpolating from keyframes.

    Args:
        frame_index: The current frame number
        keyframe_bboxes: Dict mapping keyframe index -> list of (cluster_id, bbox)
        keyframe_indices: Sorted list of keyframe indices

    Returns:
        List of (cluster_id, bbox) for this frame
    """
    # Exact keyframe match
    if frame_index in keyframe_bboxes:
        return keyframe_bboxes[frame_index]

    if not keyframe_indices:
        return []

    # Find surrounding keyframes
    prev_idx = None
    next_idx = None
    for ki in keyframe_indices:
        if ki <= frame_index:
            prev_idx = ki
        if ki > frame_index and next_idx is None:
            next_idx = ki

    # Before first or after last keyframe
    if prev_idx is None and next_idx is not None:
        return keyframe_bboxes[next_idx]
    if next_idx is None and prev_idx is not None:
        return keyframe_bboxes[prev_idx]
    if prev_idx is None or next_idx is None:
        return []

    # Interpolate
    t = (frame_index - prev_idx) / (next_idx - prev_idx)
    prev_faces = keyframe_bboxes[prev_idx]
    next_faces = keyframe_bboxes[next_idx]

    # Match faces by cluster_id
    prev_by_cluster = {cid: bbox for cid, bbox in prev_faces}
    next_by_cluster = {cid: bbox for cid, bbox in next_faces}

    result = []
    all_clusters = set(prev_by_cluster.keys()) | set(next_by_cluster.keys())
    for cid in all_clusters:
        if cid in prev_by_cluster and cid in next_by_cluster:
            bbox = interpolate_bboxes(prev_by_cluster[cid], next_by_cluster[cid], t)
            result.append((cid, bbox))
        elif cid in prev_by_cluster:
            # Face leaving frame — use prev bbox (fade out handled by caller if desired)
            result.append((cid, prev_by_cluster[cid]))
        else:
            # Face entering frame — use next bbox
            result.append((cid, next_by_cluster[cid]))

    return result
