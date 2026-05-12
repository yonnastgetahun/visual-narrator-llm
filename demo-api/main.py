from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse

from vn.ad import (
    AD_COST_PER_FRAME,
    AD_DEFAULT_VOICE_ID,
    AD_TTS_COST_PER_CHAR,
    AudioGapFitMetrics,
    ELEVENLABS_MODEL,
    ELEVENLABS_TTS_URL,
    AdDescriptionError,
    AdTTSError,
    build_audio_gap_fit_summary,
    _combine_model_versions,
    generate_gap_aware_ad_narration,
    _gap_midpoint,
)
from vn.compliance import analyze_compliance
from vn.frame import FrameExtractionError, extract_frames_at
from vn.gaps import GapDetectionError, detect_gaps
from vn.score import (
    SCORE_COST_PER_FRAME,
    ScoreError,
    ScoreReport,
    _aggregate_scores,
    _detect_word_limit,
    _grade_for_score,
    _load_narrations,
    _score_description,
)
from vn.youtube import YouTubeDownloadError, download_video, is_url


ORIGINS = [
    "https://demo.vnpoverview.com",
    "https://adult.vnpoverview.com",
    "http://localhost:3000",
    "http://localhost:3001",
]
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
# 34B Nous-Hermes-2 version from the task is disabled upstream on Replicate as of 2026-05-11.
# Closest enabled LLaVA v1.6 fallback: Vicuna-13B latest version.
REPLICATE_MODEL_SLUG = "yorickvp/llava-v1.6-vicuna-13b"
REPLICATE_LLAVA_VERSION = "0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63"
REPLICATE_COST_PER_FRAME = 0.071
REPLICATE_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
REPLICATE_MAX_HTTP_RETRIES = 2
REPLICATE_RETRY_BACKOFF_SEC = 3.0
REPLICATE_MIN_REQUEST_INTERVAL_SEC = 8.0
ADULT_DEMO_SHARED_KEY_ENV = "ADULT_DEMO_SHARED_KEY"
ADULT_DEMO_VOICE_ID_ENV = "ADULT_DEMO_VOICE_ID"
ADULT_DEMO_SHARED_KEY_HEADER = "x-demo-key"
ADULT_DEMO_MAX_DURATION_SEC = 180.0
ADULT_DEMO_MAX_GAPS = 4
ADULT_DEMO_EMPTY_DESCRIPTION_RETRIES = 2
ADULT_DEMO_NEARBY_FRAME_OFFSET_SEC = 0.5
ADULT_DEMO_MAX_RUNS_PER_HOUR = 5
ADULT_DEMO_RATE_LIMIT_WINDOW_SEC = 3600.0
ADULT_DEMO_DUPLICATE_WINDOW_SEC = 900.0
SCENE_CONTEXT_SIZE = 5
SCENE_RESET_GAP_SEC = 8.0

ADULT_AD_PROMPT = (
    "You are writing audio description for an adult content video. "
    "Describe what is happening on screen in clear, direct, clinical language. "
    "Do not use euphemisms. Do not editorialize or add emotional commentary. "
    "Focus on physical actions, positions, and visible participants. "
    "Keep the description to 2-3 sentences that fit in a 3-5 second narration gap. "
    "Write in present tense."
)
DIRECT_VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".webm", ".mkv"}
ADULT_DEMO_RECENT_RUNS: dict[str, deque[float]] = defaultdict(deque)
ADULT_DEMO_RECENT_SOURCES: dict[str, float] = {}
ADULT_DEMO_ACTIVE_CLIENTS: set[str] = set()
ADULT_DEMO_LOCK = asyncio.Lock()
REPLICATE_REQUEST_LOCK = threading.Lock()
REPLICATE_NEXT_REQUEST_AT = 0.0


app = FastAPI(title="Visual Narrator Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScoreRequest(BaseModel):
    source: str
    manifest: dict[str, Any]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/ad")
async def stream_ad(
    request: Request,
    source: str,
    min_gap: float = 2.0,
    voice_id: str = AD_DEFAULT_VOICE_ID,
) -> EventSourceResponse:
    job_id = str(uuid4())

    async def event_generator():
        try:
            if not is_url(source):
                yield _event("error", {"message": "Source must be a valid http or https URL."})
                return

            with tempfile.TemporaryDirectory(prefix="vn-demo-ad-") as tmp:
                tmp_root = Path(tmp)
                video_root = tmp_root / "video"

                yield _event("step", {"step": "download", "message": "Downloading video..."})
                video_path = await run_in_threadpool(_download_source_video, source, video_root)
                if await request.is_disconnected():
                    return

                yield _event("step", {"step": "gaps", "message": "Detecting narration gaps..."})
                gaps = await run_in_threadpool(detect_gaps, video_path, min_gap)
                compliance = await run_in_threadpool(analyze_compliance, video_path, min_gap, gaps)
                yield _event(
                    "step",
                    {
                        "step": "gaps_done",
                        "gaps": len(gaps),
                        "duration_seconds": round(compliance.total_duration_sec, 3),
                    },
                )
                if await request.is_disconnected():
                    return

                narrations: list[dict[str, Any]] = []
                audio_gap_fit_metrics: list[AudioGapFitMetrics] = []
                model_versions: list[str] = []
                context_window: deque[str] = deque(maxlen=SCENE_CONTEXT_SIZE)
                resolved_voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ADAM") or AD_DEFAULT_VOICE_ID
                total = len(gaps)

                for current, gap in enumerate(gaps, start=1):
                    if gap.duration_sec > SCENE_RESET_GAP_SEC:
                        context_window.clear()
                    midpoint = _gap_midpoint(gap)
                    frame_timestamps = _ad_frame_timestamps(gap)
                    yield _event(
                        "step",
                        {
                            "step": "describing",
                            "current": current,
                            "total": total,
                            "message": f"Describing gap {current} of {total}...",
                        },
                    )
                    frames = await run_in_threadpool(
                        extract_frames_at,
                        video_path,
                        frame_timestamps,
                        tmp_root / "frames" / f"{current:05d}",
                    )
                    yield _event(
                        "step",
                        {
                            "step": "synthesizing",
                            "current": current,
                            "total": total,
                            "message": f"Synthesizing audio {current} of {total}...",
                        },
                    )
                    narration = await run_in_threadpool(
                        generate_gap_aware_ad_narration,
                        [frame.path for frame in frames],
                        gap.duration_sec,
                        resolved_voice_id,
                        _scene_context_prefix(context_window),
                    )
                    if narration.model_version:
                        model_versions.append(narration.model_version)
                    audio_gap_fit_metrics.append(narration.audio_fit)
                    context_window.append(narration.description)
                    narrations.append(
                        {
                            "srt_index": current,
                            "start_sec": round(gap.start_sec, 3),
                            "end_sec": round(gap.end_sec, 3),
                            "gap_duration_sec": round(gap.duration_sec, 3),
                            "gap_type": gap.gap_type,
                            "frame_timestamp_sec": round(midpoint, 3),
                            "description": narration.description,
                            "audio_data": base64.b64encode(narration.audio_bytes).decode("ascii"),
                            "audio_mime": "audio/mpeg",
                            "audio_duration_sec": round(narration.audio_duration_sec, 3),
                            "audio_fit": narration.audio_fit.json_dict(),
                            "gpt_cost": round(AD_COST_PER_FRAME, 6),
                            "tts_cost": round(narration.character_count * AD_TTS_COST_PER_CHAR, 6),
                        }
                    )
                    if await request.is_disconnected():
                        return

                compliance = await run_in_threadpool(
                    analyze_compliance,
                    video_path,
                    min_gap,
                    gaps,
                    build_audio_gap_fit_summary(audio_gap_fit_metrics),
                )
                gpt_cost_estimate = sum(item["gpt_cost"] for item in narrations)
                tts_cost_estimate = sum(item["tts_cost"] for item in narrations)
                manifest = {
                    "job_id": job_id,
                    "source": source,
                    "duration_seconds": round(compliance.total_duration_sec, 3),
                    "gaps_found": len(gaps),
                    "model_version": _combine_model_versions(model_versions),
                    "voice_id": resolved_voice_id,
                    "compliance_level": compliance.wcag_level,
                    "gpt_cost_estimate": round(gpt_cost_estimate, 6),
                    "tts_cost_estimate": round(tts_cost_estimate, 6),
                    "total_cost_estimate": round(gpt_cost_estimate + tts_cost_estimate, 6),
                    "compliance": compliance.json_dict(),
                    "narrations": narrations,
                }
                await run_in_threadpool(_persist_job_outputs, job_id, video_path, narrations, compliance)
                yield _event("complete", {"manifest": manifest})
        except Exception as exc:  # noqa: BLE001
            yield _event("error", {"message": _error_message(exc)})

    return EventSourceResponse(event_generator())


@app.get("/api/download/{job_id}/audio")
async def download_audio(job_id: str) -> FileResponse:
    path = _job_output_path(job_id, "mixed-track.mp3")
    return FileResponse(path, media_type="audio/mpeg", filename=f"{job_id}-ad-track.mp3")


@app.get("/api/download/{job_id}/srt")
async def download_srt(job_id: str) -> FileResponse:
    path = _job_output_path(job_id, "narration-track.srt")
    return FileResponse(path, media_type="application/x-subrip", filename=f"{job_id}-narration.srt")


@app.get("/api/download/{job_id}/report")
async def download_report(job_id: str) -> JSONResponse:
    path = _job_output_path(job_id, "compliance.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Stored compliance report is invalid JSON.") from exc
    return JSONResponse(content=payload)


def build_mixed_track(original_path: str, narrations: list[dict[str, Any]], output_path: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise ValueError("ffmpeg is required to build the mixed narration track.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not narrations:
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    original_path,
                    "-vn",
                    "-map",
                    "0:a:0",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
            raise ValueError(f"Unable to render the mixed narration track: {stderr}") from exc
        return

    with tempfile.TemporaryDirectory(prefix="vn-mix-") as tmp:
        tmp_root = Path(tmp)
        inputs = ["-i", original_path]
        filters: list[str] = []
        delayed_streams: list[str] = []

        for index, narration in enumerate(narrations):
            audio_data = narration.get("audio_data")
            if not audio_data:
                raise ValueError(f"Narration {index + 1} is missing audio data.")

            narration_path = tmp_root / f"narration_{index:04d}.mp3"
            narration_path.write_bytes(base64.b64decode(audio_data))

            delay_ms = max(0, round(float(narration["start_sec"]) * 1000))
            inputs += ["-i", str(narration_path)]
            delayed_label = f"n{index}"
            filters.append(f"[{index + 1}:a]adelay={delay_ms}:all=1[{delayed_label}]")
            delayed_streams.append(f"[{delayed_label}]")

        filters.append(
            f"[0:a]{''.join(delayed_streams)}amix=inputs={len(delayed_streams) + 1}:duration=first:dropout_transition=0:normalize=0[out]"
        )

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    *inputs,
                    "-filter_complex",
                    ";".join(filters),
                    "-map",
                    "[out]",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
            raise ValueError(f"Unable to render the mixed narration track: {stderr}") from exc


def build_srt(narrations: list[dict[str, Any]]) -> str:
    def format_srt_time(seconds: float) -> str:
        total_ms = max(0, round(float(seconds) * 1000))
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    return "\n\n".join(
        (
            f"{entry.get('srt_index', index)}\n"
            f"{format_srt_time(entry['start_sec'])} --> {format_srt_time(entry['end_sec'])}\n"
            f"{entry['description']}"
        )
        for index, entry in enumerate(narrations, start=1)
    )


def _persist_job_outputs(
    job_id: str,
    video_path: Path,
    narrations: list[dict[str, Any]],
    compliance: Any,
) -> None:
    job_dir = _job_output_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    build_mixed_track(str(video_path), narrations, str(job_dir / "mixed-track.mp3"))
    (job_dir / "narration-track.srt").write_text(build_srt(narrations), encoding="utf-8")
    (job_dir / "compliance.json").write_text(
        json.dumps(compliance.json_dict(), indent=2),
        encoding="utf-8",
    )


def _job_output_dir(job_id: str) -> Path:
    normalized = Path(job_id).name
    if not normalized or normalized != job_id:
        raise HTTPException(status_code=400, detail="Invalid job ID.")
    return Path("/tmp") / normalized


def _job_output_path(job_id: str, filename: str) -> Path:
    path = _job_output_dir(job_id) / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Requested job output was not found.")
    return path


def _describe_frame_for_adult_ad(frame_path: Path) -> tuple[str, str]:
    replicate_key = os.getenv("REPLICATE_API_KEY")
    if not replicate_key:
        raise RuntimeError("REPLICATE_API_KEY not set")

    with frame_path.open("rb") as handle:
        img_b64 = base64.b64encode(handle.read()).decode("ascii")
    img_suffix = frame_path.suffix.lstrip(".") or "jpeg"
    data_uri = f"data:image/{img_suffix};base64,{img_b64}"

    payload = {
        "version": REPLICATE_LLAVA_VERSION,
        "input": {
            "image": data_uri,
            "prompt": ADULT_AD_PROMPT,
            "max_tokens": 200,
            "temperature": 0.2,
        },
    }
    response: httpx.Response | None = None
    for attempt in range(REPLICATE_MAX_HTTP_RETRIES + 1):
        _pace_replicate_request()
        response = httpx.post(
            REPLICATE_API_URL,
            json=payload,
            headers={
                "Authorization": f"Token {replicate_key}",
                "Content-Type": "application/json",
                "Prefer": "wait",
            },
            timeout=90,
        )
        if response.status_code not in REPLICATE_RETRYABLE_STATUS_CODES or attempt >= REPLICATE_MAX_HTTP_RETRIES:
            break

        retry_after = response.headers.get("retry-after")
        try:
            delay = max(float(retry_after), REPLICATE_RETRY_BACKOFF_SEC)
        except (TypeError, ValueError):
            delay = REPLICATE_RETRY_BACKOFF_SEC * (attempt + 1)
        time.sleep(delay)

    assert response is not None
    response.raise_for_status()
    data = response.json()
    output = data.get("output") or []
    description = "".join(output).strip() if isinstance(output, list) else str(output).strip()
    if not description:
        raise AdDescriptionError("Replicate returned an empty frame description.")
    return description, f"replicate/{REPLICATE_MODEL_SLUG.split('/', 1)[1]}@{REPLICATE_LLAVA_VERSION[:8]}"


def _pace_replicate_request() -> None:
    global REPLICATE_NEXT_REQUEST_AT

    with REPLICATE_REQUEST_LOCK:
        now = time.time()
        wait_seconds = max(0.0, REPLICATE_NEXT_REQUEST_AT - now)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.time()
        REPLICATE_NEXT_REQUEST_AT = now + REPLICATE_MIN_REQUEST_INTERVAL_SEC


def _ad_frame_timestamps(gap: Any) -> list[float]:
    midpoint = _gap_midpoint(gap)
    if gap.duration_sec <= 1.0:
        return [midpoint, midpoint, midpoint]

    start_frame = min(gap.start_sec + 0.5, midpoint)
    end_frame = max(gap.end_sec - 0.5, midpoint)
    return [start_frame, midpoint, end_frame]


def _scene_context_prefix(context_window: deque[str]) -> str | None:
    if not context_window:
        return None
    joined = "\n".join(context_window)
    return (
        "Prior descriptions in this scene:\n"
        f"{joined}\n\n"
        "Continue in the same style. Do not re-introduce characters already described."
    )


def _adult_demo_retry_timestamp(gap: Any) -> float:
    midpoint = _gap_midpoint(gap)
    min_timestamp = gap.start_sec + 0.15
    max_timestamp = gap.end_sec - 0.15
    if min_timestamp >= max_timestamp:
        min_timestamp = gap.start_sec
        max_timestamp = gap.end_sec

    offset = min(0.75, max(0.25, min(ADULT_DEMO_NEARBY_FRAME_OFFSET_SEC, gap.duration_sec / 4)))
    candidate = midpoint + offset
    if candidate <= max_timestamp:
        return candidate

    candidate = midpoint - offset
    if candidate >= min_timestamp:
        return candidate
    return midpoint


def _describe_adult_demo_gap(video_path: Path, gap: Any, frame_root: Path) -> dict[str, Any]:
    midpoint = _gap_midpoint(gap)
    frames = extract_frames_at(video_path, [midpoint], frame_root / "primary")
    frame = frames[0]
    attempts = 0
    last_error: AdDescriptionError | None = None

    for _ in range(ADULT_DEMO_EMPTY_DESCRIPTION_RETRIES):
        attempts += 1
        try:
            description, model_version = _describe_frame_for_adult_ad(frame.path)
            return {
                "description": description,
                "model_version": model_version,
                "frame_timestamp_sec": round(frame.timestamp, 3),
                "vision_attempts": attempts,
                "skipped": False,
            }
        except AdDescriptionError as exc:
            last_error = exc

    retry_timestamp = _adult_demo_retry_timestamp(gap)
    retry_frames = extract_frames_at(video_path, [retry_timestamp], frame_root / "fallback")
    retry_frame = retry_frames[0]
    attempts += 1
    try:
        description, model_version = _describe_frame_for_adult_ad(retry_frame.path)
        return {
            "description": description,
            "model_version": model_version,
            "frame_timestamp_sec": round(retry_frame.timestamp, 3),
            "vision_attempts": attempts,
            "skipped": False,
        }
    except AdDescriptionError as exc:
        last_error = exc

    return {
        "description": None,
        "model_version": "",
        "frame_timestamp_sec": round(retry_frame.timestamp, 3),
        "vision_attempts": attempts,
        "skipped": True,
        "skip_reason": str(last_error or AdDescriptionError("Replicate returned an empty frame description.")),
    }


@app.get("/api/ad-adult")
async def stream_ad_adult(
    request: Request,
    source: str,
    min_gap: float = 2.0,
    voice_id: str = AD_DEFAULT_VOICE_ID,
) -> EventSourceResponse:
    async def event_generator():
        client_id: str | None = None
        try:
            if not os.getenv("REPLICATE_API_KEY"):
                raise RuntimeError("REPLICATE_API_KEY not set")
            if not is_url(source):
                yield _event("error", {"message": "Source must be a valid http or https URL."})
                return
            client_id = await _reserve_adult_demo_run(request, source)

            with tempfile.TemporaryDirectory(prefix="vn-demo-ad-") as tmp:
                tmp_root = Path(tmp)
                video_root = tmp_root / "video"

                yield _event("step", {"step": "download", "message": "Downloading video..."})
                video_path = await run_in_threadpool(_download_source_video, source, video_root)
                video_path = await run_in_threadpool(_prepare_adult_demo_video, video_path, tmp_root)
                if await request.is_disconnected():
                    return

                yield _event("step", {"step": "gaps", "message": "Detecting narration gaps..."})
                gaps = await run_in_threadpool(detect_gaps, video_path, min_gap)
                compliance = await run_in_threadpool(analyze_compliance, video_path, min_gap, gaps)
                selected_gaps = _select_adult_demo_gaps(gaps)
                yield _event(
                    "step",
                    {
                        "step": "gaps_done",
                        "gaps": len(selected_gaps),
                        "duration_seconds": round(compliance.total_duration_sec, 3),
                    },
                )
                if await request.is_disconnected():
                    return

                narrations: list[dict[str, Any]] = []
                model_versions: list[str] = []
                total_vision_attempts = 0
                skipped_gaps = 0
                resolved_voice_id = (
                    voice_id
                    or os.getenv(ADULT_DEMO_VOICE_ID_ENV)
                    or os.getenv("ELEVENLABS_VOICE_ADAM")
                    or AD_DEFAULT_VOICE_ID
                )
                total = len(selected_gaps)

                for current, gap in enumerate(selected_gaps, start=1):
                    yield _event(
                        "step",
                        {
                            "step": "describing",
                            "current": current,
                            "total": total,
                            "message": f"Describing gap {current} of {total}...",
                        },
                    )
                    gap_result = await run_in_threadpool(
                        _describe_adult_demo_gap,
                        video_path,
                        gap,
                        tmp_root / "frames" / f"{current:05d}",
                    )
                    total_vision_attempts += gap_result["vision_attempts"]
                    if gap_result["skipped"]:
                        skipped_gaps += 1
                        yield _event(
                            "step",
                            {
                                "step": "skipping",
                                "current": current,
                                "total": total,
                                "message": f"Skipping gap {current} of {total} after repeated empty frame descriptions.",
                            },
                        )
                        continue

                    description = gap_result["description"]
                    model_version = gap_result["model_version"]
                    if model_version:
                        model_versions.append(model_version)

                    yield _event(
                        "step",
                        {
                            "step": "synthesizing",
                            "current": current,
                            "total": total,
                            "message": f"Synthesizing audio {current} of {total}...",
                        },
                    )
                    audio_bytes, character_count = await run_in_threadpool(
                        _synthesize_speech_bytes,
                        description,
                        resolved_voice_id,
                    )
                    narrations.append(
                        {
                            "srt_index": current,
                            "start_sec": round(gap.start_sec, 3),
                            "end_sec": round(gap.end_sec, 3),
                            "gap_duration_sec": round(gap.duration_sec, 3),
                            "gap_type": gap.gap_type,
                            "frame_timestamp_sec": gap_result["frame_timestamp_sec"],
                            "description": description,
                            "audio_data": base64.b64encode(audio_bytes).decode("ascii"),
                            "audio_mime": "audio/mpeg",
                            "gpt_cost": round(gap_result["vision_attempts"] * REPLICATE_COST_PER_FRAME, 6),
                            "tts_cost": round(character_count * AD_TTS_COST_PER_CHAR, 6),
                            "vision_attempts": gap_result["vision_attempts"],
                        }
                    )
                    if await request.is_disconnected():
                        return

                if selected_gaps and not narrations:
                    raise ValueError("Adult demo could not generate descriptions for the selected gaps.")

                vision_cost_estimate = total_vision_attempts * REPLICATE_COST_PER_FRAME
                tts_cost_estimate = sum(item["tts_cost"] for item in narrations)
                manifest = {
                    "source": source,
                    "duration_seconds": round(compliance.total_duration_sec, 3),
                    "gaps_found": len(gaps),
                    "gaps_selected": len(selected_gaps),
                    "gaps_processed": len(narrations),
                    "gaps_skipped": skipped_gaps,
                    "model_version": _combine_model_versions(model_versions),
                    "voice_id": resolved_voice_id,
                    "compliance_level": compliance.wcag_level,
                    "vision_attempts": total_vision_attempts,
                    "gpt_cost_estimate": round(vision_cost_estimate, 6),
                    "vision_cost_estimate": round(vision_cost_estimate, 6),
                    "tts_cost_estimate": round(tts_cost_estimate, 6),
                    "total_cost_estimate": round(vision_cost_estimate + tts_cost_estimate, 6),
                    "cost_basis": {
                        "vision_model": REPLICATE_MODEL_SLUG,
                        "vision_version": REPLICATE_LLAVA_VERSION,
                        "vision_cost_per_prediction_usd": REPLICATE_COST_PER_FRAME,
                        "vision_pricing_unit": "per_prediction",
                        "tts_cost_per_1k_chars_usd": 0.10,
                        "tts_model": ELEVENLABS_MODEL,
                    },
                    "compliance": compliance.json_dict(),
                    "narrations": narrations,
                }
                yield _event("complete", {"manifest": manifest})
        except Exception as exc:  # noqa: BLE001
            yield _event("error", {"message": _error_message(exc)})
        finally:
            if client_id:
                await _release_adult_demo_run(client_id)

    return EventSourceResponse(event_generator())


@app.post("/api/score")
async def stream_score(request: Request, payload: ScoreRequest) -> EventSourceResponse:
    async def event_generator():
        try:
            if not is_url(payload.source):
                yield _event("error", {"message": "Source must be a valid http or https URL."})
                return

            with tempfile.TemporaryDirectory(prefix="vn-demo-score-") as tmp:
                tmp_root = Path(tmp)
                video_path = await run_in_threadpool(_download_source_video, payload.source, tmp_root / "video")
                manifest_path = tmp_root / "manifest.json"
                manifest_path.write_text(json.dumps(payload.manifest, indent=2), encoding="utf-8")

                narrations = _load_narrations(payload.manifest)
                word_limit = _detect_word_limit(payload.manifest)
                scores = []
                total = len(narrations)

                for current, narration in enumerate(narrations, start=1):
                    yield _event(
                        "step",
                        {
                            "step": "scoring",
                            "current": current,
                            "total": total,
                            "message": f"Scoring description {current} of {total}...",
                        },
                    )
                    try:
                        frames = await run_in_threadpool(
                            extract_frames_at,
                            video_path,
                            [narration.frame_timestamp_sec],
                            tmp_root / "score-frames" / f"{narration.srt_index:05d}",
                        )
                    except FrameExtractionError:
                        continue
                    score = await run_in_threadpool(
                        _score_description,
                        frames[0].path,
                        narration,
                        word_limit,
                        6.0,
                    )
                    scores.append(score)
                    if await request.is_disconnected():
                        return

                aggregate = _aggregate_scores(scores)
                report = ScoreReport(
                    source=payload.source,
                    manifest=str(manifest_path),
                    scored=len(scores),
                    flagged=sum(1 for score in scores if score.flag),
                    word_limit=word_limit,
                    aggregate=aggregate,
                    grade=_grade_for_score(aggregate.overall),
                    gpt_cost_estimate=round(sum(SCORE_COST_PER_FRAME for _ in scores), 6),
                    scores=scores,
                )
                yield _event("complete", {"report": report.json_dict()})
        except Exception as exc:  # noqa: BLE001
            yield _event("error", {"message": _error_message(exc)})

    return EventSourceResponse(event_generator())


def _event(name: str, payload: dict[str, Any]) -> dict[str, str]:
    return {"event": name, "data": json.dumps(payload)}


def _adult_demo_client_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


async def _reserve_adult_demo_run(request: Request, source_url: str) -> str:
    shared_key = os.getenv(ADULT_DEMO_SHARED_KEY_ENV)
    if not shared_key:
        raise RuntimeError(f"{ADULT_DEMO_SHARED_KEY_ENV} is not set.")
    if request.headers.get(ADULT_DEMO_SHARED_KEY_HEADER) != shared_key:
        raise PermissionError("Adult demo access key is invalid.")

    client_id = _adult_demo_client_id(request)
    now = time.time()

    async with ADULT_DEMO_LOCK:
        runs = ADULT_DEMO_RECENT_RUNS[client_id]
        while runs and now - runs[0] > ADULT_DEMO_RATE_LIMIT_WINDOW_SEC:
            runs.popleft()

        stale_sources = [
            source
            for source, timestamp in ADULT_DEMO_RECENT_SOURCES.items()
            if now - timestamp > ADULT_DEMO_DUPLICATE_WINDOW_SEC
        ]
        for stale_source in stale_sources:
            ADULT_DEMO_RECENT_SOURCES.pop(stale_source, None)

        if client_id in ADULT_DEMO_ACTIVE_CLIENTS:
            raise ValueError("Only one adult demo run can be active at a time from the same IP.")
        if len(runs) >= ADULT_DEMO_MAX_RUNS_PER_HOUR:
            raise ValueError("Adult demo rate limit reached for this IP. Try again later.")
        if source_url in ADULT_DEMO_RECENT_SOURCES:
            raise ValueError("This clip was submitted recently. Wait a few minutes before retrying the same URL.")

        runs.append(now)
        ADULT_DEMO_RECENT_SOURCES[source_url] = now
        ADULT_DEMO_ACTIVE_CLIENTS.add(client_id)

    return client_id


async def _release_adult_demo_run(client_id: str) -> None:
    async with ADULT_DEMO_LOCK:
        ADULT_DEMO_ACTIVE_CLIENTS.discard(client_id)


def _select_adult_demo_gaps(gaps: list[Any]) -> list[Any]:
    return gaps[:ADULT_DEMO_MAX_GAPS]


def _prepare_adult_demo_video(video_path: Path, tmp_root: Path) -> Path:
    _ensure_video_media_file(video_path)
    duration_seconds = _probe_media_duration(video_path)
    if duration_seconds <= ADULT_DEMO_MAX_DURATION_SEC:
        return video_path

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise ValueError("ffmpeg and ffprobe are required to trim long adult demo clips.")

    trimmed_path = tmp_root / "adult-preview.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-t",
                str(ADULT_DEMO_MAX_DURATION_SEC),
                "-c",
                "copy",
                str(trimmed_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise ValueError(f"Unable to trim adult demo clip to the first 3 minutes: {stderr}") from exc

    if not trimmed_path.exists() or trimmed_path.stat().st_size <= 0:
        raise ValueError("Unable to trim adult demo clip to the first 3 minutes.")
    return trimmed_path


def _probe_media_duration(video_path: Path) -> float:
    if shutil.which("ffprobe") is None:
        raise ValueError("ffprobe is required to inspect adult demo clips.")

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout or "{}")
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise ValueError(f"Unable to inspect adult demo clip duration: {stderr}") from exc

    format_info = payload.get("format") or {}
    duration = format_info.get("duration")
    if duration is not None:
        try:
            return float(duration)
        except (TypeError, ValueError):
            pass

    stream_durations: list[float] = []
    for stream in payload.get("streams", []):
        value = stream.get("duration")
        if value is None:
            continue
        try:
            stream_durations.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(stream_durations) if stream_durations else 0.0


def _download_source_video(source_url: str, output_dir: Path) -> Path:
    if _should_try_direct_video_download(source_url):
        try:
            video_path = _download_direct_video(source_url, output_dir)
            _ensure_video_media_file(video_path)
            return video_path
        except Exception:  # noqa: BLE001
            return download_video(source_url, output_dir)
    return download_video(source_url, output_dir)


def _should_try_direct_video_download(source_url: str) -> bool:
    if _looks_like_direct_video_url(source_url):
        return True

    for method in ("HEAD", "GET"):
        try:
            headers = {"Range": "bytes=0-0"} if method == "GET" else None
            with httpx.stream(method, source_url, headers=headers, follow_redirects=True, timeout=20.0) as response:
                response.raise_for_status()
                if _response_looks_like_video(response):
                    return True
                content_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
                if content_type.startswith("text/html"):
                    return False
        except Exception:  # noqa: BLE001
            continue

    return False


def _looks_like_direct_video_url(source_url: str) -> bool:
    parsed = urlparse(source_url)
    return Path(parsed.path).suffix.lower() in DIRECT_VIDEO_SUFFIXES


def _response_looks_like_video(response: httpx.Response) -> bool:
    content_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if content_type.startswith("video/"):
        return True
    if content_type in {"application/octet-stream", "binary/octet-stream"}:
        return True

    disposition = (response.headers.get("content-disposition") or "").lower()
    return any(suffix in disposition for suffix in DIRECT_VIDEO_SUFFIXES)


def _download_direct_video(source_url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(source_url)
    suffix = Path(parsed.path).suffix.lower() or ".mp4"
    target = output_dir / f"source{suffix}"

    try:
        with httpx.stream("GET", source_url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)
    except httpx.HTTPStatusError as exc:
        raise YouTubeDownloadError(
            f"failed to download direct video ({exc.response.status_code}): {exc.response.text[:200]}"
        ) from exc
    except httpx.RequestError as exc:
        raise YouTubeDownloadError(f"failed to download direct video: {exc}") from exc

    if not target.exists() or target.stat().st_size <= 0:
        raise YouTubeDownloadError("direct video download produced an empty file")
    return target


def _ensure_video_media_file(video_path: Path) -> None:
    if shutil.which("ffprobe") is None:
        raise ValueError("ffprobe is required to inspect adult demo clips.")

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout or "{}")
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise ValueError(f"Source URL did not resolve to a readable video file: {stderr}") from exc

    streams = payload.get("streams") or []
    if not any(stream.get("codec_type") == "video" for stream in streams):
        raise ValueError("Source URL did not resolve to a readable video file.")


def _synthesize_speech_bytes(text: str, voice_id: str) -> tuple[bytes, int]:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise AdTTSError("ELEVENLABS_API_KEY is not set.")

    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.post(
                ELEVENLABS_TTS_URL.format(voice_id=voice_id),
                json=payload,
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
            )
    except httpx.RequestError as exc:
        raise AdTTSError(f"ElevenLabs request failed: {exc}") from exc

    if response.status_code != 200:
        raise AdTTSError(f"ElevenLabs API error {response.status_code}: {response.text}")
    if not response.content:
        raise AdTTSError("ElevenLabs returned empty audio.")
    return response.content, len(text)


def _error_message(exc: Exception) -> str:
    if isinstance(
        exc,
        (
            AdDescriptionError,
            AdTTSError,
            FrameExtractionError,
            GapDetectionError,
            PermissionError,
            RuntimeError,
            ScoreError,
            YouTubeDownloadError,
            ValueError,
        ),
    ):
        return str(exc)
    return f"Unexpected error: {exc}"
