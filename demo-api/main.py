from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from vn.ad import (
    AD_COST_PER_FRAME,
    AD_DEFAULT_VOICE_ID,
    AD_TTS_COST_PER_CHAR,
    ELEVENLABS_MODEL,
    ELEVENLABS_TTS_URL,
    AdDescriptionError,
    AdTTSError,
    _combine_model_versions,
    _describe_frame_for_ad,
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

ADULT_AD_PROMPT = (
    "You are writing audio description for an adult content video. "
    "Describe what is happening on screen in clear, direct, clinical language. "
    "Do not use euphemisms. Do not editorialize or add emotional commentary. "
    "Focus on physical actions, positions, and visible participants. "
    "Keep the description to 2-3 sentences that fit in a 3-5 second narration gap. "
    "Write in present tense."
)
DIRECT_VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".webm", ".mkv"}


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
                model_versions: list[str] = []
                resolved_voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ADAM") or AD_DEFAULT_VOICE_ID
                total = len(gaps)

                for current, gap in enumerate(gaps, start=1):
                    midpoint = _gap_midpoint(gap)
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
                        [midpoint],
                        tmp_root / "frames" / f"{current:05d}",
                    )
                    description, model_version = await run_in_threadpool(_describe_frame_for_ad, frames[0].path)
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
                            "frame_timestamp_sec": round(frames[0].timestamp, 3),
                            "description": description,
                            "audio_data": base64.b64encode(audio_bytes).decode("ascii"),
                            "audio_mime": "audio/mpeg",
                            "gpt_cost": round(AD_COST_PER_FRAME, 6),
                            "tts_cost": round(character_count * AD_TTS_COST_PER_CHAR, 6),
                        }
                    )
                    if await request.is_disconnected():
                        return

                gpt_cost_estimate = sum(item["gpt_cost"] for item in narrations)
                tts_cost_estimate = sum(item["tts_cost"] for item in narrations)
                manifest = {
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
                yield _event("complete", {"manifest": manifest})
        except Exception as exc:  # noqa: BLE001
            yield _event("error", {"message": _error_message(exc)})

    return EventSourceResponse(event_generator())


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
    response.raise_for_status()
    data = response.json()
    output = data.get("output") or []
    description = "".join(output).strip() if isinstance(output, list) else str(output).strip()
    return description, f"replicate/{REPLICATE_MODEL_SLUG.split('/', 1)[1]}@{REPLICATE_LLAVA_VERSION[:8]}"


@app.get("/api/ad-adult")
async def stream_ad_adult(
    request: Request,
    source: str,
    min_gap: float = 2.0,
    voice_id: str = AD_DEFAULT_VOICE_ID,
) -> EventSourceResponse:
    if not os.getenv("REPLICATE_API_KEY"):
        raise HTTPException(status_code=500, detail="REPLICATE_API_KEY not set")

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
                model_versions: list[str] = []
                resolved_voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ADAM") or AD_DEFAULT_VOICE_ID
                total = len(gaps)

                for current, gap in enumerate(gaps, start=1):
                    midpoint = _gap_midpoint(gap)
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
                        [midpoint],
                        tmp_root / "frames" / f"{current:05d}",
                    )
                    description, model_version = await run_in_threadpool(_describe_frame_for_adult_ad, frames[0].path)
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
                            "frame_timestamp_sec": round(frames[0].timestamp, 3),
                            "description": description,
                            "audio_data": base64.b64encode(audio_bytes).decode("ascii"),
                            "audio_mime": "audio/mpeg",
                            "gpt_cost": round(AD_COST_PER_FRAME, 6),
                            "tts_cost": round(character_count * AD_TTS_COST_PER_CHAR, 6),
                        }
                    )
                    if await request.is_disconnected():
                        return

                gpt_cost_estimate = sum(item["gpt_cost"] for item in narrations)
                tts_cost_estimate = sum(item["tts_cost"] for item in narrations)
                manifest = {
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
                yield _event("complete", {"manifest": manifest})
        except Exception as exc:  # noqa: BLE001
            yield _event("error", {"message": _error_message(exc)})

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


def _download_source_video(source_url: str, output_dir: Path) -> Path:
    if _looks_like_direct_video_url(source_url):
        try:
            return _download_direct_video(source_url, output_dir)
        except Exception:  # noqa: BLE001
            return download_video(source_url, output_dir)
    return download_video(source_url, output_dir)


def _looks_like_direct_video_url(source_url: str) -> bool:
    parsed = urlparse(source_url)
    return Path(parsed.path).suffix.lower() in DIRECT_VIDEO_SUFFIXES


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
            ScoreError,
            YouTubeDownloadError,
            ValueError,
        ),
    ):
        return str(exc)
    return f"Unexpected error: {exc}"
