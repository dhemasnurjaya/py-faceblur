"""Video frame extraction module."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Frame:
    """Represents an extracted video frame."""

    path: Path
    index: int


def extract_frames(video_path: str, output_dir: str, interval: int = 30) -> List[Frame]:
    """Extract frames from video at specified interval.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        interval: Extract every Nth frame

    Returns:
        List of Frame objects
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(output_dir / "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        f"select='not(mod(n\\,{interval}))'",
        "-vsync",
        "vfr",
        "-q:v",
        "2",
        "-y",
        pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    frames = []
    for frame_path in sorted(output_dir.glob("frame_*.jpg")):
        index = int(frame_path.stem.split("_")[1])
        frames.append(Frame(path=frame_path, index=index))

    return frames
