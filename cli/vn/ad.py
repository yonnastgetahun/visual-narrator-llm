from __future__ import annotations

import json
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .api import encode_file_base64
from .compliance import ComplianceReport, analyze_compliance
from .frame import extract_frames_at
from .gaps import detect_gaps
from .output import GapResult, render_ad_srt


AD_PROMPT = (
    "You are writing a professional audio description for accessibility compliance (WCAG 2.1 Level AA).\n"
    "Describe the key visual information concisely and objectively. Focus on:\n"
    "- Actions and movements relevant to understanding the content\n"
    "- Text that appears on screen\n"
    "- Changes in scene or setting\n"
    "- Expressions or gestures that convey meaning not audible in the dialogue\n"
    "Do not describe irrelevant background details. Do not interpret emotions unless clearly expressed.\n"
    "Use present tense. One to two short sentences. Maximum 30 words. Be clear and factual."
)

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
AD_DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o"
AD_COST_PER_FRAME = 0.0013
AD_TTS_COST_PER_CHAR = 0.0003


class AdDescriptionError(RuntimeError):
    """Raised when GPT-4o frame description fails."""


class AdTTSError(RuntimeError):
    """Raised when ElevenLabs speech synthesis fails."""


@dataclass(frozen=True)
class AdNarrationEntry:
    srt_index: int
    start_sec: float
    end_sec: float
    gap_duration_sec: float
    gap_type: str
    frame_timestamp_sec: float
    description: str
    audio_file: str
    gpt_cost: float
    tts_cost: float

    def json_dict(self) -> dict[str, Any]:
        return {
            "srt_index": self.srt_index,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "gap_duration_sec": round(self.gap_duration_sec, 3),
            "gap_type": self.gap_type,
            "frame_timestamp_sec": round(self.frame_timestamp_sec, 3),
            "description": self.description,
            "audio_file": self.audio_file,
            "gpt_cost": round(self.gpt_cost, 6),
            "tts_cost": round(self.tts_cost, 6),
        }


@dataclass(frozen=True)
class AdResult:
    source: str
    duration_seconds: float
    gaps_found: int
    output_dir: str
    model_version: str
    voice_id: str
    compliance_level: str
    gpt_cost_estimate: float
    tts_cost_estimate: float
    total_cost_estimate: float
    narrations: list[AdNarrationEntry]
    compliance: ComplianceReport

    def json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "duration_seconds": round(self.duration_seconds, 3),
            "gaps_found": self.gaps_found,
            "output_dir": self.output_dir,
            "model_version": self.model_version,
            "voice_id": self.voice_id,
            "compliance_level": self.compliance_level,
            "gpt_cost_estimate": round(self.gpt_cost_estimate, 6),
            "tts_cost_estimate": round(self.tts_cost_estimate, 6),
            "total_cost_estimate": round(self.total_cost_estimate, 6),
            "compliance": self.compliance.json_dict(),
            "narrations": [narration.json_dict() for narration in self.narrations],
        }


def assemble_ad_kit(
    source: Path,
    min_gap: float = 2.0,
    voice_id: str | None = None,
    source_label: str | None = None,
    output_dir: Path | None = None,
) -> AdResult:
    requested_output_dir = output_dir or Path("./vn-ad-output")
    resolved_output_dir = requested_output_dir.expanduser()
    audio_dir = resolved_output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    gaps = detect_gaps(source, min_gap=min_gap)
    timestamps = [_gap_midpoint(gap) for gap in gaps]
    narrations: list[AdNarrationEntry] = []
    model_versions: list[str] = []
    resolved_voice_id = (
        voice_id
        or os.getenv("ELEVENLABS_VOICE_ADAM")
        or AD_DEFAULT_VOICE_ID
    )

    if timestamps:
        with tempfile.TemporaryDirectory(prefix="vn-ad-frames-") as tmp:
            frame_dir = Path(tmp)
            frames = extract_frames_at(source, timestamps, frame_dir)
            for index, (gap, frame) in enumerate(zip(gaps, frames, strict=True), start=1):
                description, model_version = _describe_frame_for_ad(frame.path)
                if model_version:
                    model_versions.append(model_version)

                audio_filename = f"{index:05d}_{int(gap.start_sec * 1000):07d}.mp3"
                audio_relative_path = Path("audio") / audio_filename
                character_count = _synthesize_speech(
                    description,
                    resolved_voice_id,
                    resolved_output_dir / audio_relative_path,
                )
                narrations.append(
                    AdNarrationEntry(
                        srt_index=index,
                        start_sec=gap.start_sec,
                        end_sec=gap.end_sec,
                        gap_duration_sec=gap.duration_sec,
                        gap_type=gap.gap_type,
                        frame_timestamp_sec=frame.timestamp,
                        description=description,
                        audio_file=audio_relative_path.as_posix(),
                        gpt_cost=AD_COST_PER_FRAME,
                        tts_cost=character_count * AD_TTS_COST_PER_CHAR,
                    )
                )

    compliance = analyze_compliance(source, min_gap=min_gap, gaps=gaps)
    gpt_cost_estimate = sum(narration.gpt_cost for narration in narrations)
    tts_cost_estimate = sum(narration.tts_cost for narration in narrations)
    result = AdResult(
        source=source_label or str(source),
        duration_seconds=round(compliance.total_duration_sec, 3),
        gaps_found=len(gaps),
        output_dir=_display_output_dir(requested_output_dir),
        model_version=_combine_model_versions(model_versions),
        voice_id=resolved_voice_id,
        compliance_level=compliance.wcag_level,
        gpt_cost_estimate=gpt_cost_estimate,
        tts_cost_estimate=tts_cost_estimate,
        total_cost_estimate=gpt_cost_estimate + tts_cost_estimate,
        narrations=narrations,
        compliance=compliance,
    )

    manifest_path = resolved_output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(result.json_dict(), indent=2), encoding="utf-8")
    srt_content = render_ad_srt(result)
    (resolved_output_dir / "narration.srt").write_text(srt_content, encoding="utf-8")
    return result


def _describe_frame_for_ad(frame_path: Path) -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AdDescriptionError("OPENAI_API_KEY is not set.")

    mime_type = mimetypes.guess_type(frame_path.name)[0] or "image/jpeg"
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": AD_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encode_file_base64(frame_path)}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 150,
    }

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.post(
                OPENAI_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise AdDescriptionError(
            f"OpenAI API error {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise AdDescriptionError(f"OpenAI request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise AdDescriptionError(f"OpenAI returned invalid JSON: {response.text[:300]}") from exc
    if not isinstance(data, dict):
        raise AdDescriptionError("OpenAI returned a non-object response.")

    description = _assistant_text_from_response(data).strip()
    if not description:
        raise AdDescriptionError("OpenAI returned an empty description.")
    return description, _model_version_from_response(data)


def _synthesize_speech(text: str, voice_id: str, output_path: Path) -> int:
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    if output_path.stat().st_size <= 0:
        raise AdTTSError(f"ElevenLabs returned empty audio for {output_path.name}.")
    return len(text)


def _assistant_text_from_response(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AdDescriptionError("OpenAI response did not include choices.")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise AdDescriptionError("OpenAI response did not include a valid message.")

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        return " ".join(text_parts).strip()
    raise AdDescriptionError("OpenAI response content was not text.")


def _model_version_from_response(data: dict[str, Any]) -> str:
    model = data.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return ""


def _combine_model_versions(model_versions: list[str]) -> str:
    if not model_versions:
        return OPENAI_MODEL
    unique_versions = list(dict.fromkeys(model_versions))
    if len(unique_versions) == 1:
        return unique_versions[0]
    return ", ".join(unique_versions)


def _gap_midpoint(gap: GapResult) -> float:
    return gap.start_sec + (gap.duration_sec / 2)


def _display_output_dir(path: Path) -> str:
    raw = str(path)
    if path.is_absolute():
        display = raw
    elif raw in {".", ""}:
        display = "./"
    elif raw.startswith("./") or raw.startswith("../"):
        display = raw
    else:
        display = f"./{raw}"
    return display if display.endswith("/") else f"{display}/"
