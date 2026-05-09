from __future__ import annotations

import mimetypes
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FrameExtractionError(RuntimeError):
    """Raised when frames cannot be extracted from an input source."""


@dataclass(frozen=True)
class ExtractedFrame:
    path: Path
    timestamp: float
    duration: float


def extract_frames(source: Path, output_dir: Path, fps: float = 1 / 3) -> list[ExtractedFrame]:
    if fps <= 0:
        raise FrameExtractionError("--fps must be greater than 0")
    if not source.exists():
        raise FrameExtractionError(f"input does not exist: {source}")
    if source.is_dir():
        raise FrameExtractionError(f"input must be a file, not a directory: {source}")

    interval = 1.0 / fps
    if is_image(source):
        return [ExtractedFrame(path=source, timestamp=0.0, duration=interval)]

    if shutil.which("ffmpeg") is None:
        raise FrameExtractionError("ffmpeg is required for video input. Install it with `brew install ffmpeg`.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(pattern),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise FrameExtractionError(f"ffmpeg failed to extract frames: {stderr}") from exc

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise FrameExtractionError(f"ffmpeg produced 0 frames for {source}")

    return [
        ExtractedFrame(path=frame, timestamp=index * interval, duration=interval)
        for index, frame in enumerate(frames)
    ]


def extract_frames_at(source: Path, timestamps: list[float], output_dir: Path) -> list[ExtractedFrame]:
    """Extract one frame at each specified timestamp."""
    if not source.exists():
        raise FrameExtractionError(f"input does not exist: {source}")
    if source.is_dir():
        raise FrameExtractionError(f"input must be a file, not a directory: {source}")
    if is_image(source):
        raise FrameExtractionError("gap-targeted extraction requires video input, not a single image")
    if shutil.which("ffmpeg") is None:
        raise FrameExtractionError("ffmpeg is required. Install it with `brew install ffmpeg`.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[ExtractedFrame] = []
    for index, timestamp in enumerate(timestamps):
        out_path = output_dir / f"frame_{index:06d}.jpg"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(source),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
            raise FrameExtractionError(f"ffmpeg failed at {timestamp}s: {stderr}") from exc

        if not out_path.exists():
            raise FrameExtractionError(f"ffmpeg produced no frame at {timestamp}s")

        frames.append(ExtractedFrame(path=out_path, timestamp=timestamp, duration=0.0))
    return frames


def is_image(path: Path) -> bool:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return True
    guessed_type, _ = mimetypes.guess_type(path.name)
    return bool(guessed_type and guessed_type.startswith("image/"))
