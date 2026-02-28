"""PyFaceBlur sequential CLI application."""

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

# Suppress FutureWarnings from dependencies (like scikit-image in uniface)
warnings.simplefilter(action="ignore", category=FutureWarning)

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
from .detect_yunet import YuNetDetector
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


def run() -> None:
    """Main CLI entry point."""
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        console.print(
            Panel.fit(
                "[bold blue]PyFaceBlur[/bold blue]\n\nUsage: pyfaceblur\nInteractive CLI for blurring faces in videos.",
                border_style="blue",
            )
        )
        return

    console.print(Panel.fit("[bold blue]PyFaceBlur[/bold blue]", border_style="blue"))

    # 1. Input gathering
    video_str = questionary.path(
        "Enter path to video file:",
        validate=lambda p: Path(p).expanduser().is_file() or "File does not exist",
    ).ask()

    if not video_str:
        return

    video_path = Path(video_str).expanduser()

    # Advanced settings
    use_advanced = questionary.confirm(
        "Configure advanced settings?",
        default=False,
    ).ask()

    if use_advanced is None:
        return

    # Defaults
    interval = 15
    min_cluster_size = 2
    confidence_threshold = 0.8
    min_face_size = 50
    detector_type = "retinaface"

    if use_advanced:
        interval_str = questionary.text(
            "Frame interval for face detection (default: 15):",
            default="15",
            validate=lambda text: (
                text.isdigit() and int(text) > 0 or "Must be a positive integer"
            ),
        ).ask()

        if not interval_str:
            return

        interval = int(interval_str)

        min_cluster_str = questionary.text(
            "Minimum faces to form a cluster (default: 2):",
            default="2",
            validate=lambda text: (
                text.isdigit() and int(text) >= 2 or "Must be an integer >= 2"
            ),
        ).ask()

        if not min_cluster_str:
            return

        min_cluster_size = int(min_cluster_str)

        detector_type = questionary.select(
            "Select face detector:",
            choices=[
                questionary.Choice("RetinaFace (Default)", value="retinaface"),
                questionary.Choice(
                    "YuNet (Alternative - built into OpenCV)", value="yunet"
                ),
            ],
            default="retinaface",
        ).ask()

        if not detector_type:
            return

        confidence_str = questionary.text(
            "Detection confidence threshold (0.0-1.0, default: 0.8):",
            default="0.8",
            validate=lambda text: (
                text.replace(".", "", 1).isdigit()
                and 0.0 <= float(text) <= 1.0
                or "Must be a number between 0.0 and 1.0"
            ),
        ).ask()

        if not confidence_str:
            return

        confidence_threshold = float(confidence_str)

        min_face_str = questionary.text(
            "Minimum face size in pixels (default: 50):",
            default="50",
            validate=lambda text: (
                text.isdigit() and int(text) >= 10 or "Must be an integer >= 10"
            ),
        ).ask()

        if not min_face_str:
            return

        min_face_size = int(min_face_str)

    temp_dir = tempfile.mkdtemp(prefix="pyfaceblur_")

    try:
        # 2. Processing (Extraction & Detection)
        frames_dir = str(Path(temp_dir) / "frames")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_extract = progress.add_task("[cyan]Extracting frames...", total=None)
            frames = extract_frames(str(video_path), frames_dir, interval)
            progress.update(
                task_extract,
                completed=100,
                total=100,
                description="[green]Frames extracted",
            )

            if not frames:
                console.print("[red]Error: No frames extracted.[/red]")
                return

            task_detect = progress.add_task(
                "[cyan]Detecting faces...", total=len(frames)
            )

            # Create detector based on user choice
            if detector_type == "yunet":
                detector = YuNetDetector(
                    confidence_threshold=confidence_threshold,
                    min_face_size=min_face_size,
                )
            else:
                detector = FaceDetector(
                    confidence_threshold=confidence_threshold,
                    min_face_size=min_face_size,
                )

            all_faces = []

            for i, frame in enumerate(frames):
                try:
                    faces = detector.detect_faces(frame.path, frame.index)
                    all_faces.extend(faces)
                except Exception:
                    pass
                progress.update(
                    task_detect,
                    advance=1,
                    description=f"[cyan]Detecting faces ({len(all_faces)} found)...",
                )

            detector.close()
            progress.update(task_detect, description="[green]Detection complete")

            if not all_faces:
                console.print("[yellow]No faces detected in the video.[/yellow]")
                return

            task_cluster = progress.add_task("[cyan]Clustering faces...", total=None)
            clusters = cluster_faces(all_faces, min_samples=min_cluster_size)
            real_clusters = [c for c in clusters if c.id >= 0]
            progress.update(
                task_cluster,
                completed=100,
                total=100,
                description=f"[green]Found {len(real_clusters)} people",
            )

        # 3. Face Selection
        samples_dir = Path(temp_dir) / "face_samples"
        samples_dir.mkdir(exist_ok=True)
        face_choices = []

        for cluster in real_clusters:
            best_face = max(cluster.faces, key=lambda f: f.confidence)
            image = cv2.imread(str(best_face.frame_path))
            if image is not None:
                x1, y1, x2, y2 = best_face.bbox
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    sample_path = samples_dir / f"person_{cluster.id + 1:02d}.jpg"
                    cv2.imwrite(str(sample_path), crop)

            face_choices.append(
                questionary.Choice(
                    title=f"Person {cluster.id + 1} ({len(cluster.faces)} detections)",
                    value=cluster.id,
                    checked=True,
                )
            )

        console.print("\n[bold]Face Selection[/bold]")
        console.print(
            f"Face sample images have been saved to: [blue]{samples_dir}[/blue]"
        )
        open_directory(samples_dir)
        console.print(
            "Please review the images, then select who to blur in the terminal."
        )

        if not face_choices:
            console.print("[yellow]No valid face clusters found to select.[/yellow]")
            return

        selected_cluster_ids = questionary.checkbox(
            "Select faces to blur (Space to toggle, Enter to confirm):",
            choices=face_choices,
        ).ask()

        if selected_cluster_ids is None:
            return  # User cancelled

        # Always include noise cluster (-1) if it exists
        selected_ids_set = set(selected_cluster_ids)
        for cluster in clusters:
            if cluster.id == -1:
                selected_ids_set.add(-1)

        blur_method = questionary.select(
            "Select blur method:",
            choices=[
                questionary.Choice(
                    "Pixelate (Blazingly Fast - Recommended)", value="pixelate"
                ),
                questionary.Choice("Blackout (Blazingly Fast)", value="blackout"),
                questionary.Choice("Gaussian (Moderate CPU)", value="gaussian"),
                questionary.Choice("Elliptical (Heavy CPU)", value="elliptical"),
                questionary.Choice("Median (Heavy CPU)", value="median"),
            ],
            default="pixelate",
        ).ask()

        if not blur_method:
            return

        # 4. Encoding
        console.print("\n[bold]Encoding Video[/bold]")

        # Probe early to get total frames for progress bar
        best_enc = find_best_encoder()
        encoder_name = best_enc[0]
        console.print(f"Using hardware/software encoder: [cyan]{encoder_name}[/cyan]")

        stem = video_path.stem
        suffix = video_path.suffix
        output_path = video_path.parent / f"{stem}_blurred{suffix}"

        # We will track progress via a callback
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            encode_task = progress.add_task("[cyan]Encoding...", total=100)

            def on_progress(current: int, total: int) -> None:
                if total > 0:
                    progress.update(encode_task, total=total, completed=current)

            try:
                encode_video(
                    input_path=video_path,
                    output_path=output_path,
                    clusters=clusters,
                    selected_cluster_ids=selected_ids_set,
                    frame_interval=interval,
                    blur_method=blur_method,
                    progress_callback=on_progress,
                    encoder_override=best_enc,
                )
                progress.update(encode_task, description="[green]Encoding complete!")
            except Exception as e:
                console.print(f"[red]Encoding failed: {e}[/red]")
                return

        console.print(
            f"\n[bold green]Done![/bold green] Saved to: [blue]{output_path}[/blue]"
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run()
