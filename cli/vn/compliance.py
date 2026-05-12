from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gaps import detect_gaps
from .gaps import _probe_media as probe_media
from .output import GapResult, format_gap_time


COMPLIANCE_GAP_TYPES = {"silence", "music_only"}


@dataclass(frozen=True)
class ComplianceCriterion:
    passed: bool
    level: str | None
    description: str
    metric: dict[str, Any]

    def json_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "level": self.level,
            "description": self.description,
            "metric": self.metric,
        }


@dataclass(frozen=True)
class ComplianceReport:
    score: int
    wcag_level: str
    criteria: dict[str, ComplianceCriterion]
    gaps: list[GapResult]
    recommendations: list[str]
    total_duration_sec: float
    coverage_percent: float
    max_unbroken_speech_sec: float
    audio_gap_fit: dict[str, Any] | None = None

    def json_dict(self) -> dict[str, Any]:
        payload = {
            "score": self.score,
            "wcag_level": self.wcag_level,
            "criteria": {key: value.json_dict() for key, value in self.criteria.items()},
            "gaps": [gap.json_dict() for gap in self.gaps],
            "recommendations": self.recommendations,
        }
        if self.audio_gap_fit is not None:
            payload["audio_gap_fit"] = self.audio_gap_fit
        return payload


def analyze_compliance(
    source: Path,
    min_gap: float = 2.0,
    gaps: list[GapResult] | None = None,
    audio_gap_fit: dict[str, Any] | None = None,
) -> ComplianceReport:
    """Score accessibility compliance using narration gaps from detect_gaps()."""
    resolved_gaps = gaps if gaps is not None else detect_gaps(source, min_gap=min_gap)
    duration, _has_audio = probe_media(source.expanduser().resolve())
    coverage_percent = _coverage_percent(resolved_gaps, duration)
    max_unbroken_speech_sec = _max_unbroken_speech_stretch(resolved_gaps, duration)

    criteria = {
        "wcag_1_2_3": ComplianceCriterion(
            passed=len(resolved_gaps) >= 1,
            level="A",
            description="Audio description or media alternative is available for prerecorded video.",
            metric={"narration_gaps": len(resolved_gaps)},
        ),
        "wcag_1_2_5": ComplianceCriterion(
            passed=coverage_percent >= 15.0,
            level="AA",
            description="Audio description coverage reaches at least 15% of total video duration.",
            metric={
                "coverage_percent": round(coverage_percent, 2),
                "required_percent": 15.0,
            },
        ),
        "cvaa_audio_description": ComplianceCriterion(
            passed=max_unbroken_speech_sec <= 60.0,
            level=None,
            description="No unbroken speech stretch exceeds 60 seconds.",
            metric={
                "max_unbroken_speech_sec": round(max_unbroken_speech_sec, 3),
                "limit_sec": 60.0,
            },
        ),
    }

    return ComplianceReport(
        score=_score(criteria),
        wcag_level=_wcag_level(criteria),
        criteria=criteria,
        gaps=resolved_gaps,
        recommendations=_recommendations(resolved_gaps),
        total_duration_sec=duration,
        coverage_percent=coverage_percent,
        max_unbroken_speech_sec=max_unbroken_speech_sec,
        audio_gap_fit=audio_gap_fit,
    )


def _coverage_percent(gaps: list[GapResult], duration: float) -> float:
    if duration <= 0:
        return 0.0
    covered_duration = sum(
        gap.duration_sec for gap in gaps if gap.gap_type in COMPLIANCE_GAP_TYPES
    )
    return covered_duration / duration * 100


def _max_unbroken_speech_stretch(gaps: list[GapResult], duration: float) -> float:
    if duration <= 0:
        return 0.0

    max_stretch = 0.0
    cursor = 0.0
    for gap in sorted(gaps, key=lambda item: (item.start_sec, item.end_sec)):
        start = max(0.0, min(duration, gap.start_sec))
        end = max(0.0, min(duration, gap.end_sec))
        if start > cursor:
            max_stretch = max(max_stretch, start - cursor)
        cursor = max(cursor, end)

    if cursor < duration:
        max_stretch = max(max_stretch, duration - cursor)
    return max_stretch


def _score(criteria: dict[str, ComplianceCriterion]) -> int:
    passed = sum(1 for criterion in criteria.values() if criterion.passed)
    return round(passed / len(criteria) * 100)


def _wcag_level(criteria: dict[str, ComplianceCriterion]) -> str:
    if all(criterion.passed for criterion in criteria.values()):
        return "AA"
    if criteria["wcag_1_2_3"].passed:
        return "A"
    return "fails"


def _recommendations(gaps: list[GapResult]) -> list[str]:
    return [
        f"Add narration at {format_gap_time(gap.start_sec)} \u2014 {gap.duration_sec:.1f}s available"
        for gap in gaps[:10]
    ]
