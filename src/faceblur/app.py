"""PyFaceBlur sequential CLI application."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Set model cache before any uniface imports
os.environ.setdefault(
    "UNIFACE_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "models"),
)

from .cluster import cluster_faces
from .detect import FaceDetector
from .encode import encode_video, find_best_encoder
from .video import extract_frames

console = Console()


def open_directory(path: Path) -> None:
    """Open a directory in the system's default file manager."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform == "win32":
            os.startfile(str(path))
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        console.print(f"[yellow]Could not automatically open directory: {e}[/yellow]")
