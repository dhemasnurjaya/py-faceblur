"""Face detection module using UniFace (RetinaFace + ArcFace)."""

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

from uniface.detection import RetinaFace
from uniface.recognition import ArcFace


@dataclass
class FaceData:
    """Detected face with embedding."""

    id: int
    frame_path: Path
    frame_index: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    embedding: np.ndarray
    confidence: float
    landmarks: np.ndarray = field(default_factory=lambda: np.empty(0))


class FaceDetector:
    """Face detector using RetinaFace + ArcFace via UniFace.

    Supports multi-scale detection to catch faces at different distances,
    and filters out low-quality detections based on face size.

    Key design: Detection runs at multiple scales, but embedding extraction
    ALWAYS uses the original image to ensure consistent embeddings for clustering.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        min_face_size: int = 50,
        scales: Optional[List[float]] = None,
    ):
        """Initialize the face detector.

        Args:
            confidence_threshold: Minimum confidence to accept a detection (default: 0.8)
            min_face_size: Minimum face width/height in pixels for reliable embeddings (default: 50)
            scales: List of image scales to run detection on (default: [1.0, 1.5])
                    - 1.0: Normal scale for regular faces
                    - 1.5: Upscaled to catch small/distant faces
        """
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        # Simplified scales: 1.0 (normal) + 1.5 (catch small faces)
        # Removed 0.5x as it rarely helps and can cause issues
        self.scales = scales or [1.0, 1.5]

        self.detector = RetinaFace(confidence_threshold=confidence_threshold)
        self.recognizer = ArcFace()

    def _nms_boxes(
        self, boxes: List[Tuple], scores: List[float], iou_threshold: float = 0.4
    ) -> List[int]:
        """Non-maximum suppression to remove duplicate detections from multi-scale.

        Args:
            boxes: List of (x1, y1, x2, y2) bounding boxes
            scores: Confidence scores for each box
            iou_threshold: IOU threshold for suppression (lower = more aggressive)

        Returns:
            List of indices to keep
        """
        if not boxes:
            return []

        boxes_arr = np.array(boxes, dtype=np.float32)
        scores_arr = np.array(scores, dtype=np.float32)

        x1, y1, x2, y2 = (
            boxes_arr[:, 0],
            boxes_arr[:, 1],
            boxes_arr[:, 2],
            boxes_arr[:, 3],
        )
        areas = (x2 - x1) * (y2 - y1)

        order = scores_arr.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect_faces(self, frame_path: Path, frame_index: int) -> List[FaceData]:
        """Detect faces in a frame using multi-scale detection and generate embeddings.

        Runs detection at multiple image scales to catch faces at different distances,
        then applies NMS to remove duplicates. Embeddings are ALWAYS extracted from
        the original image to ensure consistency for clustering.

        Args:
            frame_path: Path to the frame image
            frame_index: Index of the frame in the video

        Returns:
            List of FaceData objects with bboxes, embeddings, and confidence
        """
        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"Could not read image: {frame_path}")

        h, w = image.shape[:2]

        # Collect detections from all scales
        all_detections = []  # (bbox, confidence, landmarks_on_original)

        for scale in self.scales:
            if scale == 1.0:
                scaled_image = image
            else:
                new_w, new_h = int(w * scale), int(h * scale)
                if new_w < 100 or new_h < 100:
                    continue  # Skip if scaled image is too small
                scaled_image = cv2.resize(image, (new_w, new_h))

            detections = self.detector.detect(scaled_image)

            for det in detections:
                # Scale bbox and landmarks back to original image coordinates
                if scale != 1.0:
                    x1, y1, x2, y2 = det.bbox
                    bbox = (
                        int(x1 / scale),
                        int(y1 / scale),
                        int(x2 / scale),
                        int(y2 / scale),
                    )
                    landmarks = det.landmarks / scale
                else:
                    x1, y1, x2, y2 = det.bbox
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    landmarks = det.landmarks.copy()

                all_detections.append((bbox, det.confidence, landmarks))

        if not all_detections:
            return []

        # Apply NMS to remove duplicates from multi-scale detection
        # Using lower IOU threshold (0.4) to be more aggressive at removing duplicates
        boxes = [d[0] for d in all_detections]
        scores = [d[1] for d in all_detections]
        keep_indices = self._nms_boxes(boxes, scores, iou_threshold=0.4)

        # Filter and generate embeddings
        faces = []
        face_idx = 0

        for idx in keep_indices:
            bbox_tuple, confidence, landmarks = all_detections[idx]
            x1, y1, x2, y2 = bbox_tuple

            # Filter by minimum face size for reliable embeddings
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < self.min_face_size or face_h < self.min_face_size:
                continue

            # Clamp bbox to image boundaries
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            bbox: Tuple[int, int, int, int] = (x1, y1, x2, y2)

            # Clamp landmarks to image boundaries
            landmarks = landmarks.copy()
            landmarks[:, 0] = np.clip(landmarks[:, 0], 0, w - 1)
            landmarks[:, 1] = np.clip(landmarks[:, 1], 0, h - 1)

            # IMPORTANT: Always extract embedding from ORIGINAL image
            # This ensures consistent embeddings regardless of detection scale
            try:
                embedding = self.recognizer.get_normalized_embedding(image, landmarks)
                embedding = embedding.flatten()
            except Exception:
                # Skip faces where embedding extraction fails
                continue

            faces.append(
                FaceData(
                    id=frame_index * 1000 + face_idx,
                    frame_path=frame_path,
                    frame_index=frame_index,
                    bbox=bbox,
                    embedding=embedding,
                    confidence=confidence,
                    landmarks=landmarks,
                )
            )
            face_idx += 1

        return faces

    def close(self):
        """Release resources."""
        pass
