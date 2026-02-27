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
from textual.containers import Center, Middle, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Static
from textual_autocomplete import PathAutoComplete


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
        width: 60;
        height: auto;
        max-height: 24;
        border: round $accent;
        padding: 1 2;
    }

    #logo {
        text-align: center;
        color: $text;
        text-style: bold;
        margin-bottom: 1;
    }

    #video-input {
        margin-bottom: 1;
    }

    #interval-input {
        width: 20;
        margin-bottom: 1;
    }

    #start-btn {
        margin-top: 1;
        width: 100%;
    }

    #error-label {
        color: $error;
        text-align: center;
        display: none;
    }

    .field-label {
        margin-bottom: 0;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical(id="app-container"):
                    yield Static(LOGO, id="logo")
                    yield Label("Video file:", classes="field-label")
                    video_input = Input(
                        placeholder="Enter path to video file...",
                        id="video-input",
                    )
                    yield video_input
                    yield PathAutoComplete(target=video_input, path=".")
                    yield Label("Frame interval:", classes="field-label")
                    yield Input(
                        value="30",
                        placeholder="Frame interval",
                        id="interval-input",
                        type="integer",
                    )
                    yield Label("", id="error-label")
                    yield Button("Start", id="start-btn", variant="primary")

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


# Forward declaration — ProcessingScreen will be added in Task 5
class ProcessingScreen(Screen):
    """Placeholder — replaced in Task 5."""

    def __init__(self, video_path: Path, interval: int) -> None:
        super().__init__()
        self.video_path = video_path
        self.interval = interval

    def compose(self) -> ComposeResult:
        yield Label(f"Processing {self.video_path}...")


class PyFaceBlurApp(App):
    """PyFaceBlur TUI application."""

    TITLE = "PyFaceBlur"
    CSS = """
    Screen {
        background: $surface;
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
