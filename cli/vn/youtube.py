from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yt_dlp


class YouTubeDownloadError(RuntimeError):
    """Raised when a YouTube URL cannot be downloaded."""


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def download_video(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "source.%(ext)s")
    options = {
        "format": "bv*[height<=720]+ba/b[height<=720]/best",
        "merge_output_format": "mp4",
        "outtmpl": template,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            downloaded = Path(downloader.prepare_filename(info))
    except Exception as exc:
        raise YouTubeDownloadError(f"failed to download video: {exc}") from exc

    mp4_path = downloaded.with_suffix(".mp4")
    if mp4_path.exists():
        return mp4_path
    if downloaded.exists():
        return downloaded

    candidates = sorted(output_dir.glob("source.*"))
    if candidates:
        return candidates[0]
    raise YouTubeDownloadError("yt-dlp completed but no downloaded file was found")
