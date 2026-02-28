"""Face detection module using YuNet (OpenCV built-in) + ArcFace.

YuNet is a lightweight face detector built into OpenCV 4.5.4+.
It has good accuracy with fewer false positives than some other detectors.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

from uniface.recognition import ArcFace

from .detect import FaceData


class YuNetDetector:
    """Face detector using YuNet (OpenCV) + ArcFace for embeddings.

    YuNet is a lightweight CNN-based face detector that provides:
    - Good accuracy with fewer false positives
    - 5-point facial landmarks for alignment
    - Built into OpenCV, no additional dependencies

    Supports multi-scale detection to catch faces at different distances.
    """

    # Default model path relative to package
    DEFAULT_MODEL = "models/face_detection_yunet_2023mar.onnx"

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        min_face_size: int = 50,
        scales: Optional[List[float]] = None,
        model_path: Optional[str] = None,
    ):
        """Initialize the YuNet face detector.

        Args:
            confidence_threshold: Minimum confidence to accept a detection (default: 0.8)
            min_face_size: Minimum face width/height in pixels (default: 50)
            scales: List of image scales for multi-scale detection (default: [1.0, 1.5])
            model_path: Path to YuNet ONNX model file (default: auto-detect)
        """
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.scales = scales or [1.0, 1.5]

        # Find model path
        if model_path is None:
            # Try relative to current working directory first
            model_path = self.DEFAULT_MODEL
            if not Path(model_path).exists():
                # Try relative to this file
                pkg_dir = Path(__file__).parent.parent.parent
                model_path = str(pkg_dir / self.DEFAULT_MODEL)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YuNet model not found at {model_path}. "
                "Please download from: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
            )

        self.model_path = model_path

        # YuNet detector will be created per-image since input size must match
        self._detector = None
        self._detector_size = None

        # ArcFace for embeddings
        self.recognizer = ArcFace()

    def _get_detector(self, width: int, height: int) -> cv2.FaceDetectorYN:
        """Get or create YuNet detector for given image size."""
        size = (width, height)
        if self._detector is None or self._detector_size != size:
            self._detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",  # config (not needed for ONNX)
                size,
                self.confidence_threshold,
                0.3,  # NMS threshold
                5000,  # top_k
            )
            self._detector_size = size
        return self._detector

    def _nms_boxes(
        self, boxes: List[Tuple], scores: List[float], iou_threshold: float = 0.4
    ) -> List[int]:
        """Non-maximum suppression to remove duplicate detections.

        Args:
            boxes: List of (x1, y1, x2, y2) bounding boxes
            scores: Confidence scores for each box
            iou_threshold: IOU threshold for suppression

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

    def _convert_yunet_landmarks(self, yunet_landmarks: np.ndarray) -> np.ndarray:
        """Convert YuNet landmarks to ArcFace order.

        YuNet order: right_eye, left_eye, nose, right_mouth, left_mouth
        ArcFace order: left_eye, right_eye, nose, mouth_left, mouth_right

        Args:
            yunet_landmarks: (5, 2) array in YuNet order

        Returns:
            (5, 2) array in ArcFace order
        """
        # Reorder: [1, 0, 2, 4, 3]
        return yunet_landmarks[[1, 0, 2, 4, 3], :]

    def detect_faces(self, frame_path: Path, frame_index: int) -> List[FaceData]:
        """Detect faces in a frame using multi-scale YuNet detection.

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
        all_detections = []  # (bbox_xyxy, confidence, landmarks_arcface_order)

        for scale in self.scales:
            if scale == 1.0:
                scaled_image = image
                scaled_w, scaled_h = w, h
            else:
                scaled_w, scaled_h = int(w * scale), int(h * scale)
                if scaled_w < 100 or scaled_h < 100:
                    continue
                scaled_image = cv2.resize(image, (scaled_w, scaled_h))

            # Get detector for this size
            detector = self._get_detector(scaled_w, scaled_h)
            _, faces = detector.detect(scaled_image)

            if faces is None:
                continue

            for face in faces:
                # YuNet output: [x, y, w, h, landmarks(10), score]
                x, y, fw, fh = face[:4]
                score = face[14]
                yunet_landmarks = face[4:14].reshape(5, 2)

                # Scale back to original coordinates
                if scale != 1.0:
                    x, y, fw, fh = x / scale, y / scale, fw / scale, fh / scale
                    yunet_landmarks = yunet_landmarks / scale

                # Convert to (x1, y1, x2, y2) format
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + fw), int(y + fh)
                bbox = (x1, y1, x2, y2)

                # Convert landmarks to ArcFace order
                arcface_landmarks = self._convert_yunet_landmarks(yunet_landmarks)

                all_detections.append((bbox, float(score), arcface_landmarks))

        if not all_detections:
            return []

        # Apply NMS to remove duplicates from multi-scale detection
        boxes = [d[0] for d in all_detections]
        scores = [d[1] for d in all_detections]
        keep_indices = self._nms_boxes(boxes, scores, iou_threshold=0.4)

        # Filter and generate embeddings
        faces = []
        face_idx = 0

        for idx in keep_indices:
            bbox_tuple, confidence, landmarks = all_detections[idx]
            x1, y1, x2, y2 = bbox_tuple

            # Filter by minimum face size
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

            # Extract embedding from original image using ArcFace
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
        self._detector = None
