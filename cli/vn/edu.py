from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .api import encode_file_base64
from .compliance import ComplianceReport, analyze_compliance
from .frame import extract_frames_at
from .gaps import detect_gaps
from .output import GapResult


EDUCATION_PROMPT = (
    "This frame is from an educational video. Describe any visual content that supplements "
    "the spoken information: slides, diagrams, equations, code on screen, charts, graphs, or "
    "physical demonstrations. Be specific — name variables, describe chart axes, read key text "
    "visible on screen. If the frame shows only a speaker with no visual aids, respond with "
    "exactly: NO_VISUAL_AID. One to two sentences."
)

NO_VISUAL_AID_SENTINEL = "NO_VISUAL_AID"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o"
COST_PER_FRAME = 0.0013


class EduDescriptionError(RuntimeError):
    """Raised when GPT-4o frame description fails."""


@dataclass(frozen=True)
class EduNarrationEntry:
    start_sec: float
    end_sec: float
    gap_duration_sec: float
    gap_type: str
    frame_timestamp_sec: float
    description: str
    cost_estimate: float
    srt_index: int
    visual_moment: bool

    def json_dict(self) -> dict[str, Any]:
        return {
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "gap_duration_sec": round(self.gap_duration_sec, 3),
            "gap_type": self.gap_type,
            "frame_timestamp_sec": round(self.frame_timestamp_sec, 3),
            "description": self.description,
            "cost_estimate": round(self.cost_estimate, 6),
            "srt_index": self.srt_index,
            "visual_moment": self.visual_moment,
        }


@dataclass(frozen=True)
class EduKitResult:
    source: str
    duration_seconds: float
    gaps_analyzed: int
    visual_moments: int
    skipped_talking_head: int
    narrations: list[EduNarrationEntry]
    compliance: ComplianceReport
    model_version: str
    cost_estimate: float

    def json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "duration_seconds": round(self.duration_seconds, 3),
            "gaps_analyzed": self.gaps_analyzed,
            "visual_moments": self.visual_moments,
            "skipped_talking_head": self.skipped_talking_head,
            "narrations": [narration.json_dict() for narration in self.narrations],
            "compliance": self.compliance.json_dict(),
            "model_version": self.model_version,
            "cost_estimate": round(self.cost_estimate, 6),
        }


def assemble_edu_kit(
    source: Path,
    min_gap: float = 2.0,
    source_label: str | None = None,
    output_dir: Path | None = None,
) -> EduKitResult:
    gaps = detect_gaps(source, min_gap=min_gap)
    timestamps = [_gap_midpoint(gap) for gap in gaps]
    narrations: list[EduNarrationEntry] = []
    model_versions: list[str] = []

    if timestamps:
        frame_dir = output_dir or (source.parent / ".vn-edu-frames")
        frames = extract_frames_at(source, timestamps, frame_dir)
        visual_index = 0
        for gap, frame in zip(gaps, frames, strict=True):
            description, model_version = _describe_frame_for_education_response(frame.path)
            if model_version:
                model_versions.append(model_version)
            if description is None:
                continue
            visual_index += 1
            narrations.append(
                EduNarrationEntry(
                    start_sec=gap.start_sec,
                    end_sec=gap.end_sec,
                    gap_duration_sec=gap.duration_sec,
                    gap_type=gap.gap_type,
                    frame_timestamp_sec=frame.timestamp,
                    description=description,
                    cost_estimate=COST_PER_FRAME,
                    srt_index=visual_index,
                    visual_moment=True,
                )
            )

    compliance = analyze_compliance(source, min_gap=min_gap, gaps=gaps)
    visual_moments = len(narrations)

    return EduKitResult(
        source=source_label or str(source),
        duration_seconds=round(compliance.total_duration_sec, 3),
        gaps_analyzed=len(gaps),
        visual_moments=visual_moments,
        skipped_talking_head=len(gaps) - visual_moments,
        narrations=narrations,
        compliance=compliance,
        model_version=_combine_model_versions(model_versions),
        cost_estimate=len(gaps) * COST_PER_FRAME,
    )


def _describe_frame_for_education(frame_path: Path) -> str | None:
    description, _model_version = _describe_frame_for_education_response(frame_path)
    return description


def _describe_frame_for_education_response(frame_path: Path) -> tuple[str | None, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EduDescriptionError("OPENAI_API_KEY is not set.")

    mime_type = mimetypes.guess_type(frame_path.name)[0] or "image/jpeg"
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EDUCATION_PROMPT},
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
        raise EduDescriptionError(
            f"OpenAI API error {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise EduDescriptionError(f"OpenAI request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise EduDescriptionError(f"OpenAI returned invalid JSON: {response.text[:300]}") from exc
    if not isinstance(data, dict):
        raise EduDescriptionError("OpenAI returned a non-object response.")

    description = _assistant_text_from_response(data)
    normalized = description.strip()
    if _is_no_visual_aid(normalized):
        return None, _model_version_from_response(data)
    if not normalized:
        raise EduDescriptionError("OpenAI returned an empty description.")
    return normalized, _model_version_from_response(data)


def _assistant_text_from_response(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise EduDescriptionError("OpenAI response did not include choices.")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise EduDescriptionError("OpenAI response did not include a valid message.")

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        return " ".join(text_parts).strip()
    raise EduDescriptionError("OpenAI response content was not text.")


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


def _is_no_visual_aid(description: str) -> bool:
    normalized = description.strip().strip(" \t\r\n.!?:;\"'()[]{}")
    return bool(normalized) and normalized.upper() == NO_VISUAL_AID_SENTINEL


def _gap_midpoint(gap: GapResult) -> float:
    return gap.start_sec + (gap.duration_sec / 2)
