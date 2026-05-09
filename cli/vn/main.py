from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import typer

from .api import DEFAULT_API_URL, VNApiError, VNClient
from .compliance import analyze_compliance
from .frame import FrameExtractionError, extract_frames
from .gaps import GapDetectionError, detect_gaps
from .output import render_compliance_report, render_gap_results, render_results, result_from_api_response
from .youtube import YouTubeDownloadError, download_video, is_url


app = typer.Typer(help="Visual Narrator command line tools.")
keys_app = typer.Typer(help="Manage Visual Narrator API keys.")
app.add_typer(keys_app, name="keys")

OutputFormat = typer.Option("json", "--format", "-f", help="Output format: json, srt, or text.")
ApiUrl = typer.Option(DEFAULT_API_URL, "--api-url", help="Visual Narrator API base URL.")
WhisperModel = typer.Option("base", "--whisper-model", help="Whisper model to use for gap detection.")


@app.command()
def describe(
    source: str = typer.Argument(..., help="Local video/image file or YouTube URL."),
    output_format: str = OutputFormat,
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Visual Narrator API key. Defaults to VN_API_KEY."),
    api_url: str = ApiUrl,
    fps: float = typer.Option(1 / 3, "--fps", min=0.001, help="Frame sampling rate. Default is 0.333, one frame every 3 seconds."),
) -> None:
    """Generate audio descriptions with timecodes."""
    output_format = _normalize_format(output_format)
    resolved_api_key = api_key or os.getenv("VN_API_KEY")
    if not resolved_api_key:
        _fail("Missing API key. Pass --api-key or set VN_API_KEY.")

    client = VNClient(api_url=api_url, api_key=resolved_api_key)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            frames = extract_frames(media_path, tmp_path / "frames", fps=fps)
            results = []
            for frame in frames:
                response = client.describe_frame(frame.path)
                results.append(result_from_api_response(response, frame.timestamp, frame.duration))
        except (FrameExtractionError, YouTubeDownloadError, VNApiError) as exc:
            _fail(str(exc))

    typer.echo(render_results(results, output_format))


@app.command()
def benchmark(
    image_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="JPEG or PNG image file."),
    api_url: str = ApiUrl,
) -> None:
    """Run a single-frame VN vs GPT-4o vs Gemini benchmark."""
    client = VNClient(api_url=api_url)
    try:
        result = client.benchmark_frame(image_file)
    except VNApiError as exc:
        _fail(str(exc))
    typer.echo(json.dumps(result, indent=2))


@app.command()
def gaps(
    source: str = typer.Argument(..., help="Local video file or YouTube URL."),
    output_format: str = OutputFormat,
    min_gap: float = typer.Option(2.0, "--min-gap", min=0.001, help="Filter out gaps shorter than this many seconds."),
    whisper_model: str = WhisperModel,
) -> None:
    """Detect narration-friendly dialogue gaps with Whisper."""
    output_format = _normalize_format(output_format)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            gaps = detect_gaps(media_path, whisper_model=whisper_model, min_gap=min_gap)
        except (GapDetectionError, YouTubeDownloadError) as exc:
            _fail(str(exc))

    typer.echo(render_gap_results(gaps, output_format))


@app.command()
def compliance(
    source: str = typer.Argument(..., help="Local video file or YouTube URL."),
    output_format: str = typer.Option("json", "--format", "-f", help="Output format: json or text."),
    min_gap: float = typer.Option(2.0, "--min-gap", min=0.001, help="Filter out gaps shorter than this many seconds."),
    whisper_model: str = WhisperModel,
) -> None:
    """Generate a WCAG/CVAA compliance report from detected narration gaps."""
    output_format = _normalize_compliance_format(output_format)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            report = analyze_compliance(media_path, whisper_model=whisper_model, min_gap=min_gap)
        except (GapDetectionError, YouTubeDownloadError) as exc:
            _fail(str(exc))

    typer.echo(render_compliance_report(report, output_format))


@keys_app.command("create")
def keys_create(
    email: str = typer.Argument(..., help="Email address for the free-tier API key."),
    api_url: str = ApiUrl,
) -> None:
    """Create a free-tier Visual Narrator API key."""
    client = VNClient(api_url=api_url)
    try:
        result = client.create_key(email)
    except VNApiError as exc:
        _fail(str(exc))
    typer.echo(json.dumps(result, indent=2))


def _resolve_source(source: str, download_dir: Path) -> Path:
    if is_url(source):
        return download_video(source, download_dir)
    return Path(source).expanduser().resolve()


def _normalize_format(output_format: str) -> str:
    normalized = output_format.lower()
    if normalized not in {"json", "srt", "text"}:
        _fail("--format must be one of: json, srt, text")
    return normalized


def _normalize_compliance_format(output_format: str) -> str:
    normalized = output_format.lower()
    if normalized not in {"json", "text"}:
        _fail("--format must be one of: json, text")
    return normalized


def _fail(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)
