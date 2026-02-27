"""Face detection module using UniFace (RetinaFace + ArcFace)."""

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

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
    """Face detector using RetinaFace + ArcFace via UniFace."""

    def __init__(self, confidence_threshold: float = 0.7):
        self.detector = RetinaFace(confidence_threshold=confidence_threshold)
        self.recognizer = ArcFace()

    def detect_faces(self, frame_path: Path, frame_index: int) -> List[FaceData]:
        """Detect faces in a frame and generate embeddings.

        Args:
            frame_path: Path to the frame image
            frame_index: Index of the frame in the video

        Returns:
            List of FaceData objects with bboxes, embeddings, and confidence
        """
        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"Could not read image: {frame_path}")

        detections = self.detector.detect(image)

        faces = []
        for i, det in enumerate(detections):
            bbox = tuple(int(v) for v in det.bbox)  # (x1, y1, x2, y2)
            confidence = det.confidence
            landmarks = det.landmarks

            embedding = self.recognizer.get_normalized_embedding(image, landmarks)
            embedding = embedding.flatten()

            faces.append(
                FaceData(
                    id=frame_index * 100 + i,
                    frame_path=frame_path,
                    frame_index=frame_index,
                    bbox=bbox,
                    embedding=embedding,
                    confidence=confidence,
                    landmarks=landmarks,
                )
            )

        return faces

    def close(self):
        """Release resources."""
        pass
