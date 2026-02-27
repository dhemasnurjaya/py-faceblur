"""Face clustering module using DBSCAN."""

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from sklearn.cluster import DBSCAN

from .detect import FaceData


@dataclass
class Cluster:
    """A cluster of similar faces."""

    id: int
    faces: List[FaceData]


def cluster_faces(
    faces: List[FaceData], eps: float = 0.4, min_samples: int = 2
) -> List[Cluster]:
    """Cluster faces using DBSCAN based on embedding similarity.

    Args:
        faces: List of detected faces with embeddings
        eps: Maximum distance between faces in same cluster
        min_samples: Minimum faces to form a cluster

    Returns:
        List of Cluster objects
    """
    if not faces:
        return []

    embeddings = np.array([f.embedding for f in faces])

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(embeddings)

    clusters_dict: Dict[int, List[FaceData]] = {}
    for face, label in zip(faces, labels):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(face)

    clusters = []
    for label, cluster_faces in sorted(clusters_dict.items()):
        clusters.append(Cluster(id=label, faces=cluster_faces))

    return clusters
