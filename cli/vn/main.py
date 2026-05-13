from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import typer

from .ad import AdDescriptionError, AdTTSError, assemble_ad_kit
from .api import DEFAULT_API_URL, VNApiError, VNClient
from .benchmark import BenchmarkError, benchmark_video
from .compliance import analyze_compliance
from .edu import EduDescriptionError, assemble_edu_kit
from .frame import FrameExtractionError, extract_frames
from .gaps import GapDetectionError, detect_gaps
from .kit import assemble_kit
from .output import render_compliance_report, render_gap_results, render_results, result_from_api_response
from .output import render_ad, render_edu, render_kit, render_podcast, render_sports, render_theater
from .output import render_score
from .podcast import PodcastDescriptionError, PodcastMixError, PodcastTTSError, assemble_podcast
from .score import ScoreError, score_manifest
from .sports import SportsDetectionError, assemble_sports_kit
from .theater import TheaterDescriptionError, TheaterTTSError, assemble_theater_kit
from .youtube import YouTubeDownloadError, download_video, is_url


app = typer.Typer(help="Visual Narrator command line tools.")
keys_app = typer.Typer(help="Manage Visual Narrator API keys.")
app.add_typer(keys_app, name="keys")

OutputFormat = typer.Option("json", "--format", "-f", help="Output format: json, srt, or text.")
ApiUrl = typer.Option(DEFAULT_API_URL, "--api-url", help="Visual Narrator API base URL.")


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
    image_file: Optional[Path] = typer.Argument(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Legacy single-frame benchmark mode: JPEG or PNG image file.",
    ),
    video: Optional[Path] = typer.Option(
        None,
        "--video",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Video file to benchmark with the VN AD pipeline.",
    ),
    reference: Optional[Path] = typer.Option(
        None,
        "--reference",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Reference SRT file containing professional AD cues.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        dir_okay=False,
        help="Path for the benchmark JSON report.",
    ),
    voice_id: Optional[str] = typer.Option(
        None,
        "--voice-id",
        help="ElevenLabs voice ID. Defaults to ELEVENLABS_VOICE_ADAM or Adam.",
    ),
    min_gap: float = typer.Option(2.0, "--min-gap", min=0.001, help="Filter out gaps shorter than this many seconds."),
    characters: Optional[Path] = typer.Option(
        None,
        "--characters",
        dir_okay=False,
        readable=True,
        help="Character sheet .txt file (one character per line: 'Name: description'). "
             "If omitted, names are auto-detected from the audio track via Whisper.",
    ),
    no_auto_detect: bool = typer.Option(
        False,
        "--no-auto-detect",
        help="Disable automatic name detection from audio when no --characters file is provided.",
    ),
    api_url: str = ApiUrl,
) -> None:
    """Run the quality benchmark against professional AD, or a legacy single-frame benchmark."""
    if video is not None or reference is not None or output is not None:
        if image_file is not None:
            _fail("Do not pass an image file when using --video/--reference/--output.")
        if video is None or reference is None or output is None:
            _fail("--video, --reference, and --output are required together.")
        try:
            report = benchmark_video(
                video_path=video,
                reference_path=reference,
                output_path=output,
                min_gap=min_gap,
                voice_id=voice_id,
                character_sheet=characters,
                auto_detect_names=not no_auto_detect,
            )
        except (FileNotFoundError, BenchmarkError, GapDetectionError, FrameExtractionError, AdDescriptionError, AdTTSError) as exc:
            _fail(str(exc))
        typer.echo(json.dumps(report.json_dict(), indent=2))
        return

    if image_file is None:
        _fail("Pass either an image file or --video/--reference/--output.")

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
) -> None:
    """Detect narration-friendly dialogue gaps with Deepgram Nova-3."""
    output_format = _normalize_format(output_format)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            gaps = detect_gaps(media_path, min_gap=min_gap)
        except (GapDetectionError, YouTubeDownloadError) as exc:
            _fail(str(exc))

    typer.echo(render_gap_results(gaps, output_format))


@app.command()
def compliance(
    source: str = typer.Argument(..., help="Local video file or YouTube URL."),
    output_format: str = typer.Option("json", "--format", "-f", help="Output format: json or text."),
    min_gap: float = typer.Option(2.0, "--min-gap", min=0.001, help="Filter out gaps shorter than this many seconds."),
) -> None:
    """Generate a WCAG/CVAA compliance report from detected narration gaps."""
    output_format = _normalize_compliance_format(output_format)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            report = analyze_compliance(media_path, min_gap=min_gap)
        except (GapDetectionError, YouTubeDownloadError) as exc:
            _fail(str(exc))

    typer.echo(render_compliance_report(report, output_format))


@app.command()
def kit(
    source: str = typer.Argument(..., help="Local video file or YouTube URL."),
    output_format: str = OutputFormat,
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Visual Narrator API key. Defaults to VN_API_KEY."),
    api_url: str = ApiUrl,
    min_gap: float = typer.Option(2.0, "--min-gap", min=0.001, help="Filter out gaps shorter than this many seconds."),
) -> None:
    """Generate a narration script, subtitle track, and compliance report."""
    output_format = _normalize_format(output_format)
    resolved_api_key = api_key or os.getenv("VN_API_KEY")
    if not resolved_api_key:
        _fail("Missing API key. Pass --api-key or set VN_API_KEY.")

    client = VNClient(api_url=api_url, api_key=resolved_api_key)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            result = assemble_kit(
                media_path,
                client,
                min_gap=min_gap,
                source_label=source,
                output_dir=tmp_path / "kit-frames",
            )
        except (GapDetectionError, FrameExtractionError, YouTubeDownloadError, VNApiError) as exc:
            _fail(str(exc))

    typer.echo(render_kit(result, output_format))


@app.command()
def edu(
    source: str = typer.Argument(..., help="Local video file or YouTube URL."),
    output_format: str = OutputFormat,
    min_gap: float = typer.Option(2.0, "--min-gap", min=0.001, help="Filter out gaps shorter than this many seconds."),
) -> None:
    """Generate education-focused visual aid descriptions from narration gaps."""
    output_format = _normalize_format(output_format)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            result = assemble_edu_kit(
                media_path,
                min_gap=min_gap,
                source_label=source,
                output_dir=tmp_path / "edu-frames",
            )
        except (GapDetectionError, FrameExtractionError, YouTubeDownloadError, EduDescriptionError) as exc:
            _fail(str(exc))

    typer.echo(render_edu(result, output_format))


@app.command()
def sports(
    source: str = typer.Argument(..., help="Local video file or YouTube URL."),
    output_format: str = OutputFormat,
    fps: float = typer.Option(2.0, "--fps", min=0.1, help="Frame extraction rate. Default 2fps."),
    narrate_every: float = typer.Option(
        4.0,
        "--narrate-every",
        min=0.5,
        help="Seconds between narrations. Default 4.0s.",
    ),
    min_confidence: float = typer.Option(
        0.7,
        "--min-confidence",
        min=0.1,
        max=1.0,
        help="Rekognition minimum detection confidence. Default 0.7.",
    ),
) -> None:
    """Live sports play-by-play narration using Rekognition tracking + GPT-4o."""
    output_format = _normalize_format(output_format)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            result = assemble_sports_kit(
                media_path,
                fps=fps,
                narrate_every=narrate_every,
                min_confidence=min_confidence,
                source_label=source,
                output_dir=tmp_path / "sports-frames",
            )
        except (FrameExtractionError, YouTubeDownloadError, SportsDetectionError) as exc:
            _fail(str(exc))

    typer.echo(render_sports(result, output_format))


@app.command()
def theater(
    source: str = typer.Argument(..., help="Local video file or YouTube URL."),
    output_format: str = OutputFormat,
    output_dir: Path = typer.Option(
        Path("./vn-theater-output"),
        "--output-dir",
        help="Directory for MP3 files and manifest.json.",
    ),
    voice_id: Optional[str] = typer.Option(
        None,
        "--voice-id",
        help="ElevenLabs voice ID. Defaults to ELEVENLABS_VOICE_ADAM or Adam.",
    ),
    min_gap: float = typer.Option(2.0, "--min-gap", min=0.001, help="Filter out gaps shorter than this many seconds."),
) -> None:
    """Generate a timed audio description track using ElevenLabs TTS."""
    output_format = _normalize_format(output_format)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            result = assemble_theater_kit(
                media_path,
                min_gap=min_gap,
                voice_id=voice_id,
                source_label=source,
                output_dir=output_dir,
            )
        except (
            GapDetectionError,
            FrameExtractionError,
            YouTubeDownloadError,
            TheaterDescriptionError,
            TheaterTTSError,
        ) as exc:
            _fail(str(exc))

    typer.echo(render_theater(result, output_format))
    if output_format != "json":
        typer.echo(f"\nAudio output directory: {result.output_dir}")


@app.command()
def podcast(
    source: str = typer.Argument(..., help="Local video file or YouTube URL."),
    output_format: str = typer.Option("json", "--format", "-f", help="Output format: json or text."),
    output: Path = typer.Option(
        Path("./podcast-output.mp3"),
        "--output",
        help="Path for assembled MP3.",
    ),
    output_dir: Path = typer.Option(
        Path("./vn-podcast-work"),
        "--output-dir",
        help="Work directory for intermediate files.",
    ),
    voice_id: Optional[str] = typer.Option(
        None,
        "--voice-id",
        help="ElevenLabs voice ID. Defaults to ELEVENLABS_VOICE_ADAM or Adam.",
    ),
    min_gap: float = typer.Option(2.0, "--min-gap", min=0.001, help="Filter gaps shorter than this many seconds."),
) -> None:
    """Convert a video into a narrated audio podcast with original dialogue preserved."""
    output_format = _normalize_podcast_format(output_format)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            result = assemble_podcast(
                media_path,
                min_gap=min_gap,
                voice_id=voice_id,
                source_label=source,
                output_path=output,
                output_dir=output_dir,
            )
        except (
            GapDetectionError,
            FrameExtractionError,
            YouTubeDownloadError,
            PodcastDescriptionError,
            PodcastTTSError,
            PodcastMixError,
        ) as exc:
            _fail(str(exc))

    typer.echo(render_podcast(result, output_format))
    typer.echo(f"\nPodcast saved to: {result.output_file}", err=True)


@app.command()
def ad(
    source: str = typer.Argument(..., help="Local video file or YouTube URL."),
    output_format: str = OutputFormat,
    output_dir: Path = typer.Option(
        Path("./vn-ad-output"),
        "--output-dir",
        help="Directory for MP3 files, manifest.json, and narration.srt.",
    ),
    voice_id: Optional[str] = typer.Option(
        None,
        "--voice-id",
        help="ElevenLabs voice ID. Defaults to ELEVENLABS_VOICE_ADAM or Adam.",
    ),
    min_gap: float = typer.Option(2.0, "--min-gap", min=0.001, help="Filter out gaps shorter than this many seconds."),
    narrative_context: Optional[str] = typer.Option(
        None,
        "--narrative-context",
        help=(
            "Film-level narrative context injected as a system prompt prefix. "
            "Include: film title, genre, brief plot summary, character relationships, "
            "and scene context. Plain text, no special format required."
        ),
    ),
    narrative_context_file: Optional[Path] = typer.Option(
        None,
        "--narrative-context-file",
        help="Path to a plain-text file containing the narrative context.",
        exists=True,
        dir_okay=False,
    ),
    two_stage: bool = typer.Option(
        False,
        "--two-stage",
        help=(
            "Two-stage generation: dense visual description (Stage 1) → "
            "AD compression via GPT-4o mini (Stage 2) → hallucination filter (Stage 3). "
            "Requires OPENAI_API_KEY. Marginal cost: ~$0.06/film."
        ),
    ),
) -> None:
    """Generate a WCAG-compliant Standard AD track with ElevenLabs voicing."""
    output_format = _normalize_format(output_format)

    resolved_narrative_context = narrative_context
    if narrative_context_file and not narrative_context:
        resolved_narrative_context = narrative_context_file.read_text(encoding="utf-8").strip()
    elif narrative_context_file and narrative_context:
        typer.echo("Warning: both --narrative-context and --narrative-context-file provided; inline value wins.", err=True)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            result = assemble_ad_kit(
                media_path,
                min_gap=min_gap,
                voice_id=voice_id,
                source_label=source,
                output_dir=output_dir,
                narrative_context=resolved_narrative_context,
                two_stage=two_stage,
            )
        except (
            GapDetectionError,
            FrameExtractionError,
            YouTubeDownloadError,
            AdDescriptionError,
            AdTTSError,
        ) as exc:
            _fail(str(exc))

    typer.echo(render_ad(result, output_format))


@app.command()
def score(
    source: str = typer.Argument(
        ...,
        help="Local video file or YouTube URL (same source used to generate the manifest).",
    ),
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        "-m",
        help="Path to manifest.json from vn ad or vn theater.",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: json, text, or flagged.",
    ),
    output_dir: Path = typer.Option(
        Path("./vn-score-output"),
        "--output-dir",
        help="Directory for score-report.json.",
    ),
    word_limit: Optional[int] = typer.Option(
        None,
        "--word-limit",
        min=1,
        help="Word limit for within_limit check. Auto-detected from manifest if not set.",
    ),
    min_score: float = typer.Option(
        6.0,
        "--min-score",
        min=0.0,
        max=10.0,
        help="Flag threshold: descriptions with any dimension below this are flagged.",
    ),
) -> None:
    """Score AD description quality using GPT-4o Vision as a judge."""
    output_format = _normalize_score_format(output_format)

    with tempfile.TemporaryDirectory(prefix="vn-cli-") as tmp:
        tmp_path = Path(tmp)
        try:
            media_path = _resolve_source(source, tmp_path / "download")
            report = score_manifest(
                manifest_path=manifest.expanduser().resolve(),
                video_source=media_path,
                word_limit=word_limit,
                min_score=min_score,
                output_dir=output_dir,
                source_label=source,
                manifest_label=str(manifest),
            )
        except (FileNotFoundError, ValueError, FrameExtractionError, YouTubeDownloadError, ScoreError) as exc:
            _fail(str(exc))

    typer.echo(render_score(report, output_format))


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


def _normalize_podcast_format(output_format: str) -> str:
    normalized = output_format.lower()
    if normalized not in {"json", "text"}:
        _fail("--format must be one of: json, text")
    return normalized


def _normalize_score_format(output_format: str) -> str:
    normalized = output_format.lower()
    if normalized not in {"json", "text", "flagged"}:
        _fail("--format must be one of: json, text, flagged")
    return normalized


def _fail(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)
