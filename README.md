# PyFaceBlur

An interactive command-line tool that automatically detects, clusters, and blurs faces in videos. It guides you through a simple step-by-step process to extract frames, group people by facial identity, select who you want to blur, and re-encode the video.

## Features

- **Interactive CLI:** Built with `rich` and `questionary` for a clean, prompt-based UX including file path auto-completion.
- **Accurate Face Recognition:** Uses [UniFace](https://github.com/yakhyo/uniface) (RetinaFace detection + ArcFace 512-dim neural embeddings via ONNX Runtime) to accurately re-identify the same person across a video.
- **DBSCAN Clustering:** Automatically groups identical faces into "clusters" using Cosine similarity.
- **Hardware-Accelerated Encoding:** Automatically detects and leverages GPU encoders like `av1_vaapi`, `hevc_vaapi`, `h264_vaapi`, `h264_nvenc`, and more via FFmpeg.
- **Visual Face Selection:** Extracts one high-quality thumbnail per detected person and opens your system's file explorer so you can easily check boxes for who to blur.
- **Multiple Blur Styles:** Choose from Gaussian, Pixelate, Blackout, Elliptical, or Median blur methods.
- **Smooth Interpolation:** Bounding boxes are linearly interpolated between sampled keyframes and held static when faces exit/enter, ensuring smooth blurring without split-second exposures.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for fast dependency management
- `ffmpeg` installed and available in your system `$PATH` (for frame extraction and re-encoding)

## Setup

```bash
# Clone the repository and navigate to the project directory
cd py-faceblur

# Sync dependencies using uv
uv sync
```

## Usage

Run the interactive wizard:

```bash
uv run pyfaceblur
```

### The Pipeline

1. **Input:** You provide the path to your video and the frame sampling interval (e.g., sample every 30th frame).
2. **Processing:** The app uses FFmpeg to extract frames, runs RetinaFace to find all faces, and generates ArcFace embeddings.
3. **Clustering:** DBSCAN groups the embeddings to identify unique individuals.
4. **Selection:** The app saves a thumbnail of each person to a temporary folder, opens it, and asks you to select which people to blur using interactive checkboxes.
5. **Encoding:** The app finds the best available video encoder on your system, applies the chosen blur method to the selected faces, interpolates their movement, and generates a new `*_blurred.mp4` video.

## Advanced / Legacy CLI

The original proof-of-concept command-line interface is also still available for purely extracting and debugging the clustering outputs into an output folder.

```bash
uv run pyfaceblur-legacy detect --video input.mp4 --output ./output --interval 30 --confidence 0.7
```
