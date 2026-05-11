from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx
import yt_dlp


class YouTubeDownloadError(RuntimeError):
    """Raised when a YouTube URL cannot be downloaded."""


YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "m.youtube.com",
}


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_youtube_url(value: str) -> bool:
    hostname = urlparse(value).hostname
    return hostname.lower() in YOUTUBE_HOSTS if hostname else False


def _normalize_youtube_url(value: str) -> str:
    parsed = urlparse(value)
    hostname = parsed.hostname.lower() if parsed.hostname else ""

    if hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if not video_id:
            return value
        return urlunparse(
            (
                parsed.scheme or "https",
                parsed.netloc,
                "/watch",
                "",
                urlencode({"v": video_id}),
                "",
            )
        )

    if hostname == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
        if not video_id:
            return value
        return urlunparse(
            (
                parsed.scheme or "https",
                "youtu.be",
                f"/{video_id}",
                "",
                "",
                "",
            )
        )

    return value


def _download_with_yt_dlp(url: str, output_dir: Path) -> Path:
    template = str(output_dir / "source.%(ext)s")
    is_youtube = is_youtube_url(url)
    options = {
        "format": "bv*[height<=720]+ba/b[height<=720]/best",
        "merge_output_format": "mp4",
        "outtmpl": template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": is_youtube,
    }
    source_url = _normalize_youtube_url(url) if is_youtube else url

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(source_url, download=True)
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


def download_via_cobalt(source_url: str, output_dir: Path) -> Path:
    cobalt_api_url = os.getenv("COBALT_API_URL")
    if not cobalt_api_url:
        raise YouTubeDownloadError("COBALT_API_URL is not configured")

    normalized_source_url = _normalize_youtube_url(source_url) if is_youtube_url(source_url) else source_url
    request_body = {
        "url": normalized_source_url,
        "videoQuality": "360",
        "filenameStyle": "basic",
        "downloadMode": "auto",
    }

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.post(
                f"{cobalt_api_url.rstrip('/')}/",
                json=request_body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise YouTubeDownloadError(f"cobalt lookup failed: {exc}") from exc

    status = payload.get("status")
    if status not in {"redirect", "tunnel"}:
        error = payload.get("error")
        raise YouTubeDownloadError(f"cobalt returned status {status!r}: {error}")

    download_url = payload.get("url")
    if not isinstance(download_url, str) or not download_url:
        raise YouTubeDownloadError("cobalt response did not include a download url")

    destination = output_dir / "source.mp4"
    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            with client.stream("GET", download_url) as stream:
                stream.raise_for_status()
                with destination.open("wb") as file_obj:
                    for chunk in stream.iter_bytes():
                        if chunk:
                            file_obj.write(chunk)
    except httpx.HTTPError as exc:
        raise YouTubeDownloadError(f"cobalt download failed: {exc}") from exc

    if not destination.exists() or destination.stat().st_size == 0:
        raise YouTubeDownloadError("cobalt download completed without creating source.mp4")

    return destination


def download_video(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_youtube_url(url) and os.getenv("COBALT_API_URL"):
        try:
            return download_via_cobalt(url, output_dir)
        except YouTubeDownloadError:
            pass

    return _download_with_yt_dlp(url, output_dir)
