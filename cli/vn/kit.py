from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .api import VNClient
from .compliance import ComplianceReport, analyze_compliance
from .frame import extract_frames_at
from .gaps import detect_gaps
from .output import GapResult


@dataclass(frozen=True)
class NarrationEntry:
    start_sec: float
    end_sec: float
    gap_duration_sec: float
    gap_type: str
    frame_timestamp_sec: float
    description: str
    cost_estimate: float
    srt_index: int

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
        }


@dataclass(frozen=True)
class KitResult:
    source: str
    duration_seconds: float
    gaps_found: int
    narrations: list[NarrationEntry]
    compliance: ComplianceReport
    model_version: str
    cost_estimate: float

    def json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "duration_seconds": round(self.duration_seconds, 3),
            "gaps_found": self.gaps_found,
            "narrations": [narration.json_dict() for narration in self.narrations],
            "compliance": self.compliance.json_dict(),
            "model_version": self.model_version,
            "cost_estimate": round(self.cost_estimate, 6),
        }


def assemble_kit(
    source: Path,
    client: VNClient,
    min_gap: float = 2.0,
    source_label: str | None = None,
    output_dir: Path | None = None,
) -> KitResult:
    gaps = detect_gaps(source, min_gap=min_gap)
    timestamps = [_gap_midpoint(gap) for gap in gaps]
    narrations: list[NarrationEntry] = []
    model_versions: list[str] = []

    if timestamps:
        frame_dir = output_dir or (source.parent / ".vn-kit-frames")
        frames = extract_frames_at(source, timestamps, frame_dir)
        for index, (gap, frame) in enumerate(zip(gaps, frames, strict=True), start=1):
            response = client.describe_frame(frame.path)
            description = _description_from_response(response)
            cost_estimate = _cost_from_response(response)
            model_version = _model_version_from_response(response)
            if model_version:
                model_versions.append(model_version)
            narrations.append(
                NarrationEntry(
                    start_sec=gap.start_sec,
                    end_sec=gap.end_sec,
                    gap_duration_sec=gap.duration_sec,
                    gap_type=gap.gap_type,
                    frame_timestamp_sec=frame.timestamp,
                    description=description,
                    cost_estimate=cost_estimate,
                    srt_index=index,
                )
            )

    compliance = analyze_compliance(source, min_gap=min_gap, gaps=gaps)
    total_cost = sum(narration.cost_estimate for narration in narrations)

    return KitResult(
        source=source_label or str(source),
        duration_seconds=round(compliance.total_duration_sec, 3),
        gaps_found=len(gaps),
        narrations=narrations,
        compliance=compliance,
        model_version=_combine_model_versions(model_versions),
        cost_estimate=total_cost,
    )


def _gap_midpoint(gap: GapResult) -> float:
    return gap.start_sec + (gap.duration_sec / 2)


def _description_from_response(response: dict[str, Any]) -> str:
    description = response.get("description") or response.get("narration") or ""
    return str(description).strip()


def _cost_from_response(response: dict[str, Any]) -> float:
    raw_cost = response.get("cost_estimate")
    if raw_cost is None:
        performance = response.get("performance")
        if isinstance(performance, dict):
            raw_cost = performance.get("cost_estimate")
    try:
        return float(raw_cost)
    except (TypeError, ValueError):
        return 0.0


def _model_version_from_response(response: dict[str, Any]) -> str:
    for key in ("model_version", "model", "version"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    analysis = response.get("analysis")
    if isinstance(analysis, dict):
        for key in ("model_version", "model", "version"):
            value = analysis.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _combine_model_versions(model_versions: list[str]) -> str:
    if not model_versions:
        return ""
    unique_versions = list(dict.fromkeys(model_versions))
    if len(unique_versions) == 1:
        return unique_versions[0]
    return ", ".join(unique_versions)
