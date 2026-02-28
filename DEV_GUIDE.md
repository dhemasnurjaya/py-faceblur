# PyFaceBlur Developer & AI Guide

This document provides architectural context, data flow maps, and strict guidelines for human developers and AI coding assistants working on the `py-faceblur` codebase.

> **Note to AI Assistants:** Read this entire document before proposing architectural changes or adding dependencies to this project.

## Architecture & Data Flow

The project is structured as a linear, sequential CLI pipeline. It extracts frames, detects faces, clusters them by identity, asks the user for input, and re-encodes the video.

### 1. Module Responsibilities

- **`src/faceblur/app.py`**: The main entry point. Orchestrates the sequential CLI flow using `rich` for UI/progress and `questionary` for user prompts. 
- **`src/faceblur/video.py`**: Uses `ffmpeg` (via subprocess) to extract frames from the input video at a specified interval.
- **`src/faceblur/detect.py`**: Wraps the **UniFace** library. Uses RetinaFace for bounding boxes and landmarks, and ArcFace for generating 512-dimensional facial embeddings.
- **`src/faceblur/cluster.py`**: Takes the 512-dim embeddings and runs **DBSCAN** (from `scikit-learn`) using the **cosine distance** metric to group identical faces into clusters.
- **`src/faceblur/blur.py`**: Contains the image manipulation logic. Responsible for drawing blurs (Gaussian, pixelate, etc.) and handling the linear interpolation of bounding boxes between keyframes.
- **`src/faceblur/encode.py`**: Probes the video, auto-detects hardware encoders, and pipes raw BGR bytes from OpenCV to an `ffmpeg` subprocess to re-encode the video with blurs applied.
- **`src/faceblur/cli.py`**: The legacy/debug CLI entry point. It runs the pipeline but dumps raw frames and face crops to disk instead of re-encoding a video.

### 2. The Data Pipeline

1. **Extraction**: Video -> Every Nth Frame saved to `/tmp/.../frames/`
2. **Detection**: Frame -> `FaceData` dataclass (contains `id`, `bbox (x1, y1, x2, y2)`, `embedding`, `landmarks`).
3. **Clustering**: List of `FaceData` -> List of `Cluster` dataclasses.
4. **Interpolation (Encoding)**: 
   - A lookup table of keyframes is built.
   - For frames between keyframes, bounding boxes are **linearly interpolated**.
   - If a face enters or leaves, the bounding box is held **static** to the nearest known location to prevent split-second privacy exposures.
5. **Muxing**: OpenCV reads frames -> Blur applied -> Piped as `rawvideo` to FFmpeg -> FFmpeg copies original audio and encodes video.

## Core Dependencies

- **UniFace (`uniface`)**: The core AI engine for detection and recognition. Do not swap this out for MediaPipe, dlib, or plain OpenCV Haar cascades. It uses ONNX runtime under the hood.
- **FFmpeg**: Required on the host system. Used for both initial frame extraction and final video re-encoding.
- **OpenCV (`opencv-python`)**: Used exclusively for image reading/writing and drawing the actual blur arrays. *Not* used for video demuxing/muxing (FFmpeg handles that).
- **scikit-learn**: Used purely for the `DBSCAN` clustering algorithm.
- **Rich & Questionary**: Used for the terminal user interface.

## Guidelines & Strict Rules for AI Agents

When modifying this codebase, adhere to the following rules:

1. **Hardware Encoding Fallback is Sacred**:
   In `encode.py`, the `find_best_encoder()` function meticulously probes the system for `av1_vaapi`, `hevc_vaapi`, `h264_vaapi`, `h264_nvenc`, etc., using a `1280x720` test video. Do not simplify or break this logic. If you modify it, ensure hardware encoders are prioritized over CPU (`libopenh264`).

2. **Bounding Box Format**:
   Bounding boxes are strictly formatted as `(x1, y1, x2, y2)`. Do not attempt to use `(x, y, w, h)` anywhere in the pipeline.

3. **Dependency Discipline**:
   Do not add heavy dependencies (like PyTorch or TensorFlow) unless explicitly requested by the user. UniFace running on ONNX is specifically chosen to keep the footprint light (~50MB).

4. **UI Paradigm**:
   The CLI uses a synchronous, sequential prompt flow (`questionary`). Do not introduce asynchronous UI frameworks (like Textual) or background worker threads. The application blocks while waiting for user input, and blocks while processing with a `rich` progress bar.

5. **Temp File Cleanup**:
   The pipeline generates gigabytes of raw frames in a temp directory during execution. Ensure any new exit paths or error states properly trigger the `shutil.rmtree(temp_dir)` cleanup block in `app.py`.