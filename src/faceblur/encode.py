"""Video re-encoding with face blur applied."""

import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .blur import BlurMethod, apply_blur, get_bboxes_for_frame
from .cluster import Cluster
from .detect import FaceData


ENCODER_PRIORITY = [
    "h264_nvenc",
    "h264_vaapi",
    "h264_amf",
    "h264_qsv",
    "libopenh264",
]


def probe_video(video_path: Path) -> dict:
    """Probe video file to get codec, bitrate, fps, resolution, audio info.

    Args:
        video_path: Path to input video

    Returns:
        Dict with keys: width, height, fps, bitrate, codec, audio_codec, audio_bitrate
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    info = {
        "width": 0,
        "height": 0,
        "fps": 30.0,
        "bitrate": "4M",
        "codec": "h264",
        "audio_codec": None,
        "audio_bitrate": None,
        "total_frames": 0,
    }

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            info["width"] = int(stream.get("width", 0))
            info["height"] = int(stream.get("height", 0))
            info["codec"] = stream.get("codec_name", "h264")

            # Parse fps from r_frame_rate (e.g., "60/1")
            fps_str = stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                info["fps"] = float(num) / float(den) if float(den) > 0 else 30.0
            else:
                info["fps"] = float(fps_str)

            # Bitrate from stream or format
            if stream.get("bit_rate"):
                info["bitrate"] = stream["bit_rate"]
            elif data.get("format", {}).get("bit_rate"):
                info["bitrate"] = data["format"]["bit_rate"]

            # Total frames
            nb_frames = stream.get("nb_frames")
            if nb_frames and nb_frames != "N/A":
                info["total_frames"] = int(nb_frames)
            else:
                duration = float(data.get("format", {}).get("duration", 0))
                info["total_frames"] = int(duration * info["fps"])

        elif stream.get("codec_type") == "audio":
            info["audio_codec"] = stream.get("codec_name")
            info["audio_bitrate"] = stream.get("bit_rate")

    return info


def find_best_encoder() -> str:
    """Find the best available H.264 encoder by testing each in priority order.

    Returns:
        Name of the best available encoder
    """
    for encoder in ENCODER_PRIORITY:
        cmd = [
            "ffmpeg",
            "-v",
            "quiet",
            "-f",
            "lavfi",
            "-i",
            "nullsrc=s=64x64:d=0.1",
            "-c:v",
            encoder,
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode == 0:
            return encoder

    raise RuntimeError(
        "No H.264 encoder found. Available encoders checked: "
        + ", ".join(ENCODER_PRIORITY)
    )


def build_keyframe_bboxes(
    clusters: List[Cluster],
    selected_cluster_ids: Set[int],
    frame_interval: int,
) -> Tuple[Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]], List[int]]:
    """Build a lookup of keyframe bboxes for selected clusters.

    Args:
        clusters: All clusters from detection
        selected_cluster_ids: Set of cluster IDs to blur
        frame_interval: The interval used for frame extraction

    Returns:
        (keyframe_bboxes dict, sorted keyframe_indices list)
    """
    keyframe_bboxes: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = {}

    for cluster in clusters:
        if cluster.id not in selected_cluster_ids:
            continue
        for face in cluster.faces:
            # Convert 1-based frame file index to 0-based video frame index
            video_frame_idx = (face.frame_index - 1) * frame_interval
            if video_frame_idx not in keyframe_bboxes:
                keyframe_bboxes[video_frame_idx] = []
            keyframe_bboxes[video_frame_idx].append((cluster.id, face.bbox))

    keyframe_indices = sorted(keyframe_bboxes.keys())
    return keyframe_bboxes, keyframe_indices


def encode_video(
    input_path: Path,
    output_path: Path,
    clusters: List[Cluster],
    selected_cluster_ids: Set[int],
    frame_interval: int,
    blur_method: BlurMethod = "gaussian",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Re-encode video with face blur applied to selected clusters.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        clusters: All detected clusters
        selected_cluster_ids: Cluster IDs to blur
        frame_interval: Frame interval used during detection
        blur_method: Blur method to use
        progress_callback: Called with (current_frame, total_frames)
    """
    video_info = probe_video(input_path)
    encoder = find_best_encoder()

    keyframe_bboxes, keyframe_indices = build_keyframe_bboxes(
        clusters,
        selected_cluster_ids,
        frame_interval,
    )

    width = video_info["width"]
    height = video_info["height"]
    fps = video_info["fps"]
    bitrate = video_info["bitrate"]
    total_frames = video_info["total_frames"]

    # Build FFmpeg encode command
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
    ]

    # Map audio from original if present
    if video_info["audio_codec"]:
        ffmpeg_cmd.extend(["-map", "1:a:0", "-c:a", "copy"])

    ffmpeg_cmd.extend(
        [
            "-c:v",
            encoder,
            "-b:v",
            str(bitrate),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )

    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    # Start FFmpeg process
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get bboxes for this frame (exact or interpolated)
            face_bboxes = get_bboxes_for_frame(
                frame_idx,
                keyframe_bboxes,
                keyframe_indices,
            )

            # Apply blur to each face
            for _cluster_id, bbox in face_bboxes:
                frame = apply_blur(frame, bbox, method=blur_method)

            # Write frame to FFmpeg
            proc.stdin.write(frame.tobytes())

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx, total_frames)

    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"FFmpeg encoding failed: {stderr}")
