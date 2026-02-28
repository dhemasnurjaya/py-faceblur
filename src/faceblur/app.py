"""PyFaceBlur TUI application."""

import os
from pathlib import Path

# Set model cache before any uniface imports
os.environ.setdefault(
    "UNIFACE_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "models"),
)

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Middle, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
)
from textual.events import Key
import glob


class PathInput(Input):
    """Custom input for bash-like path tab completion."""

    def on_key(self, event: Key) -> None:
        if event.key == "tab":
            current = self.value
            if not current:
                return

            path = os.path.expanduser(current)
            matches = glob.glob(path + "*")

            if os.path.isdir(path) and not current.endswith("/"):
                matches = glob.glob(path + "/*")
                if not matches:
                    self.value = current + "/"
                    self.cursor_position = len(self.value)
                    event.prevent_default()
                    event.stop()
                    return

            if not matches:
                return

            if len(matches) == 1:
                match = matches[0]
                if os.path.isdir(match):
                    match += "/"
            else:
                match = os.path.commonprefix(matches)

            if current.startswith("~"):
                match = match.replace(os.path.expanduser("~"), "~", 1)

            if match != current:
                self.value = match
                self.cursor_position = len(self.value)
                event.prevent_default()
                event.stop()


LOGO = r"""
 ____        _____              ____  _
|  _ \ _   _|  ___|_ _  ___ __|  _ \| |_   _ _ __
| |_) | | | | |_ / _` |/ __/ _ \ |_) | | | | | '__|
|  __/| |_| |  _| (_| | (_|  __/  _ <| | |_| | |
|_|    \__, |_|  \__,_|\___\___|_| \_\_|\__,_|_|
       |___/
"""


class WelcomeScreen(Screen):
    """Welcome screen with video path input and settings."""

    DEFAULT_CSS = """
    WelcomeScreen {
        align: center middle;
    }

    #app-container {
        width: 50;
        height: auto;
        padding: 1 2;
    }

    #logo {
        text-align: center;
        color: $accent;
        text-style: bold;
        margin-bottom: 2;
        width: 100%;
    }

    .form-input {
        margin-bottom: 1;
        width: 100%;
    }

    #start-btn {
        width: 100%;
        margin-top: 1;
    }

    #error-label {
        color: $error;
        text-align: center;
        display: none;
        width: 100%;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical(id="app-container"):
                    yield Label("PyFaceBlur", id="logo")

                    yield PathInput(
                        placeholder="Path to video file...",
                        id="video-input",
                        classes="form-input",
                    )

                    yield Input(
                        value="30",
                        placeholder="Frame interval (default 30)",
                        id="interval-input",
                        type="integer",
                        classes="form-input",
                    )

                    yield Label("", id="error-label")
                    yield Button("Start Processing", id="start-btn", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn":
            self._validate_and_start()

    def _validate_and_start(self) -> None:
        video_path = self.query_one("#video-input", Input).value.strip()
        interval_str = self.query_one("#interval-input", Input).value.strip()
        error_label = self.query_one("#error-label", Label)

        if not video_path:
            error_label.update("Please enter a video file path.")
            error_label.styles.display = "block"
            return

        path = Path(video_path)
        if not path.exists():
            error_label.update(f"File not found: {video_path}")
            error_label.styles.display = "block"
            return

        try:
            interval = int(interval_str)
            if interval < 1:
                raise ValueError
        except ValueError:
            error_label.update("Frame interval must be a positive integer.")
            error_label.styles.display = "block"
            return

        error_label.styles.display = "none"
        self.app.push_screen(ProcessingScreen(video_path=path, interval=interval))


class ProcessingScreen(Screen):
    """Processing screen — extract, detect, cluster with progress."""

    DEFAULT_CSS = """
    ProcessingScreen {
        align: center middle;
    }

    #processing-container {
        width: 60;
        height: auto;
        padding: 1 2;
    }

    #processing-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #phase-label {
        margin-bottom: 0;
    }

    #status-label {
        color: $text-muted;
        margin-top: 0;
    }

    #progress-bar {
        margin: 1 0;
    }
    """

    def __init__(self, video_path: Path, interval: int) -> None:
        super().__init__()
        self.video_path = video_path
        self.interval = interval
        self.frames = []
        self.all_faces = []
        self.clusters = []

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical(id="processing-container"):
                    yield Static("PyFaceBlur", id="processing-title")
                    yield Label("[1/3] Preparing...", id="phase-label")
                    yield ProgressBar(total=100, id="progress-bar")
                    yield Label("", id="status-label")

    def on_mount(self) -> None:
        self.run_worker(self._process, thread=True)

    def _update_ui(self, phase: str, status: str, progress: float) -> None:
        """Thread-safe UI update."""
        self.app.call_from_thread(self._do_update_ui, phase, status, progress)

    def _do_update_ui(self, phase: str, status: str, progress: float) -> None:
        self.query_one("#phase-label", Label).update(phase)
        self.query_one("#status-label", Label).update(status)
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(progress=progress)

    def _process(self) -> None:
        """Run the full detection pipeline in a background thread."""
        from .video import extract_frames
        from .detect import FaceDetector
        from .cluster import cluster_faces

        # Phase 1: Extract frames
        self._update_ui("[1/3] Extracting frames...", "Starting FFmpeg...", 0)
        import tempfile

        self._temp_dir = tempfile.mkdtemp(prefix="pyfaceblur_")
        frames_dir = str(Path(self._temp_dir) / "frames")

        self.frames = extract_frames(
            str(self.video_path),
            frames_dir,
            self.interval,
        )
        self._update_ui(
            "[1/3] Extracting frames...",
            f"Extracted {len(self.frames)} frames",
            33,
        )

        if not self.frames:
            self._update_ui("[1/3] Error", "No frames extracted from video.", 0)
            return

        # Phase 2: Detect faces
        self._update_ui("[2/3] Detecting faces...", "Initializing detector...", 33)
        detector = FaceDetector()
        self.all_faces = []

        for i, frame in enumerate(self.frames):
            try:
                faces = detector.detect_faces(frame.path, frame.index)
                self.all_faces.extend(faces)
            except Exception:
                pass
            progress = 33 + (i + 1) / len(self.frames) * 34
            self._update_ui(
                "[2/3] Detecting faces...",
                f"Frame {i + 1}/{len(self.frames)} — {len(self.all_faces)} faces found",
                progress,
            )

        detector.close()

        if not self.all_faces:
            self._update_ui("[2/3] Error", "No faces detected in video.", 67)
            return

        # Phase 3: Cluster
        self._update_ui("[3/3] Clustering faces...", "Running DBSCAN...", 67)
        self.clusters = cluster_faces(self.all_faces)
        self._update_ui(
            "[3/3] Clustering complete",
            f"Found {len([c for c in self.clusters if c.id >= 0])} people",
            100,
        )

        # Write face samples and transition to selection screen
        face_samples = self._write_face_samples()
        self.app.call_from_thread(
            self.app.push_screen,
            FaceSelectionScreen(
                video_path=self.video_path,
                interval=self.interval,
                clusters=self.clusters,
                all_faces=self.all_faces,
                face_samples=face_samples,
                temp_dir=self._temp_dir,
            ),
        )

    def _write_face_samples(self) -> dict:
        """Write one representative face crop per cluster. Returns {cluster_id: path}."""
        import cv2

        samples_dir = Path(self._temp_dir) / "face_samples"
        samples_dir.mkdir(exist_ok=True)
        face_samples = {}

        for cluster in self.clusters:
            if cluster.id < 0:
                continue
            # Pick the face with highest confidence as representative
            best_face = max(cluster.faces, key=lambda f: f.confidence)
            image = cv2.imread(str(best_face.frame_path))
            if image is None:
                continue
            x1, y1, x2, y2 = best_face.bbox
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            sample_path = samples_dir / f"person_{cluster.id:02d}.jpg"
            cv2.imwrite(str(sample_path), crop)
            face_samples[cluster.id] = sample_path

        return face_samples


class FaceSelectionScreen(Screen):
    """Face selection screen — choose which faces to blur."""

    DEFAULT_CSS = """
    FaceSelectionScreen {
        align: center middle;
    }

    #selection-container {
        width: 65;
        height: auto;
        padding: 1 2;
    }

    #faces-list {
        height: auto;
        max-height: 20;
        overflow-y: auto;
        margin-bottom: 1;
        border: solid $accent-muted;
        padding: 1;
    }

    #selection-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #selection-help {
        color: $text-muted;
        margin-bottom: 1;
    }

    .face-row {
        height: 3;
        margin-bottom: 0;
        align: left middle;
    }

    .face-checkbox {
        width: 1fr;
    }

    .view-btn {
        width: auto;
        margin-left: 2;
    }

    #blur-method-row {
        height: 3;
        align: left middle;
        margin-top: 1;
    }

    #blur-method-label {
        width: 16;
        content-align: right middle;
        margin-right: 1;
        color: $text-muted;
    }

    #blur-method {
        width: 24;
    }

    #buttons-row {
        margin-top: 1;
        height: 3;
        align: center middle;
    }

    #buttons-row Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        video_path: Path,
        interval: int,
        clusters: list,
        all_faces: list,
        face_samples: dict,
        temp_dir: str,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.interval = interval
        self.clusters = clusters
        self.all_faces = all_faces
        self.face_samples = face_samples
        self.temp_dir = temp_dir
        # Only show real clusters (not noise cluster -1)
        self.real_clusters = [c for c in clusters if c.id >= 0]

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical(id="selection-container"):
                    yield Static("PyFaceBlur — Select Faces", id="selection-title")
                    yield Label(
                        f"Found {len(self.real_clusters)} people. "
                        "All faces will be blurred.\n"
                        "Uncheck faces you want to KEEP visible.",
                        id="selection-help",
                    )

                    with Vertical(id="faces-list"):
                        for cluster in self.real_clusters:
                            sample_info = ""
                            if cluster.id in self.face_samples:
                                sample_info = (
                                    f"  [dim]{self.face_samples[cluster.id]}[/dim]"
                                )
                            with Horizontal(classes="face-row"):
                                yield Checkbox(
                                    f"Person {cluster.id + 1}  "
                                    f"({len(cluster.faces)} detections)"
                                    f"{sample_info}",
                                    value=True,
                                    id=f"cluster-{cluster.id}",
                                    classes="face-checkbox",
                                )
                                if cluster.id in self.face_samples:
                                    yield Button(
                                        "View",
                                        id=f"view-{cluster.id}",
                                        variant="default",
                                        classes="view-btn",
                                    )

                    with Horizontal(id="blur-method-row"):
                        yield Label("Blur method:", id="blur-method-label")
                        yield Select(
                            [
                                ("Gaussian", "gaussian"),
                                ("Pixelate", "pixelate"),
                                ("Blackout", "blackout"),
                                ("Elliptical", "elliptical"),
                                ("Median", "median"),
                            ],
                            value="gaussian",
                            id="blur-method",
                        )
                    with Horizontal(id="buttons-row"):
                        yield Button("Back", id="back-btn")
                        yield Button(
                            "Blur Selected",
                            id="blur-btn",
                            variant="primary",
                        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "blur-btn":
            self._start_encoding()
        elif event.button.id and event.button.id.startswith("view-"):
            cluster_id = int(event.button.id.split("-")[1])
            if cluster_id in self.face_samples:
                self._open_image(self.face_samples[cluster_id])

    def _open_image(self, path: Path) -> None:
        """Open image in default viewer."""
        import subprocess
        import sys

        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)])
            elif sys.platform == "win32":
                os.startfile(str(path))
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception:
            pass  # Ignore if viewer fails to launch

    def _start_encoding(self) -> None:
        # Collect selected cluster IDs
        selected_ids = set()
        for cluster in self.real_clusters:
            cb = self.query_one(f"#cluster-{cluster.id}", Checkbox)
            if cb.value:
                selected_ids.add(cluster.id)

        # Always include noise cluster (-1) if it exists
        for cluster in self.clusters:
            if cluster.id == -1:
                selected_ids.add(-1)

        blur_method = str(self.query_one("#blur-method", Select).value)
        if blur_method == "Select.NoSelection":
            blur_method = "gaussian"

        # Generate output path
        stem = self.video_path.stem
        suffix = self.video_path.suffix
        output_path = self.video_path.parent / f"{stem}_blurred{suffix}"

        self.app.push_screen(
            EncodingScreen(
                video_path=self.video_path,
                output_path=output_path,
                clusters=self.clusters,
                selected_cluster_ids=selected_ids,
                interval=self.interval,
                blur_method=blur_method,
                temp_dir=self.temp_dir,
            )
        )


class EncodingScreen(Screen):
    """Encoding screen — re-encode video with face blur applied."""

    DEFAULT_CSS = """
    EncodingScreen {
        align: center middle;
    }

    #encoding-container {
        width: 60;
        height: auto;
        padding: 1 2;
    }

    #encoding-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #encoding-status {
        color: $text-muted;
    }

    #encoding-progress {
        margin: 1 0;
    }

    #output-label {
        margin-top: 1;
    }

    #done-btn {
        margin-top: 1;
        width: 100%;
        display: none;
    }
    """

    def __init__(
        self,
        video_path: Path,
        output_path: Path,
        clusters: list,
        selected_cluster_ids: set,
        interval: int,
        blur_method: str,
        temp_dir: str,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.clusters = clusters
        self.selected_cluster_ids = selected_cluster_ids
        self.interval = interval
        self.blur_method = blur_method
        self.temp_dir = temp_dir

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical(id="encoding-container"):
                    yield Static("PyFaceBlur", id="encoding-title")
                    yield Label("Encoding video with face blur...", id="encoding-phase")
                    yield ProgressBar(total=100, id="encoding-progress")
                    yield Label("", id="encoding-status")
                    yield Label(f"Output: {self.output_path.name}", id="output-label")
                    yield Button("Done — Exit", id="done-btn", variant="success")

    def on_mount(self) -> None:
        self.run_worker(self._encode, thread=True)

    def _update_ui(self, status: str, progress: float) -> None:
        self.app.call_from_thread(self._do_update_ui, status, progress)

    def _do_update_ui(self, status: str, progress: float) -> None:
        self.query_one("#encoding-status", Label).update(status)
        bar = self.query_one("#encoding-progress", ProgressBar)
        bar.update(progress=progress)

    def _encode(self) -> None:
        """Run encoding in background thread."""
        from .encode import encode_video, find_best_encoder

        def on_progress(current: int, total: int) -> None:
            pct = (current / total) * 100 if total > 0 else 0
            self._update_ui(f"Frame {current}/{total} (Encoder: {encoder_name})", pct)

        try:
            best_enc = find_best_encoder()
            encoder_name = best_enc[0]
            self._update_ui(f"Starting encoding with {encoder_name}...", 0)

            encoder_used = encode_video(
                input_path=self.video_path,
                output_path=self.output_path,
                clusters=self.clusters,
                selected_cluster_ids=self.selected_cluster_ids,
                frame_interval=self.interval,
                blur_method=self.blur_method,
                progress_callback=on_progress,
                encoder_override=best_enc,
            )
            self._update_ui(f"Encoding complete! (Used {encoder_used})", 100)
            self.app.call_from_thread(self._show_done)
        except Exception as e:
            self._update_ui(f"Error: {e}", 0)

    def _show_done(self) -> None:
        self.query_one("#encoding-phase", Label).update("Encoding complete!")
        btn = self.query_one("#done-btn", Button)
        btn.styles.display = "block"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "done-btn":
            # Cleanup temp directory
            import shutil

            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.app.exit()


class PyFaceBlurApp(App):
    """PyFaceBlur TUI application."""

    TITLE = "PyFaceBlur"
    CSS = """
    Screen {
        background: $surface;
        padding: 1 2;
    }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
    ]

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())


def run() -> None:
    """Entry point for the TUI app."""
    app = PyFaceBlurApp()
    app.run()
