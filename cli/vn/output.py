from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DescriptionResult:
    timecode: str
    description: str
    objects_detected: list[Any]
    latency_ms: int | None
    start_seconds: float
    duration_seconds: float

    def json_dict(self) -> dict[str, Any]:
        return {
            "timecode": self.timecode,
            "description": self.description,
            "objects_detected": self.objects_detected,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class GapResult:
    start_sec: float
    end_sec: float
    duration_sec: float
    gap_type: str

    def json_dict(self) -> dict[str, Any]:
        return {
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "gap_type": self.gap_type,
        }


def render_json(results: Iterable[DescriptionResult]) -> str:
    return json.dumps([result.json_dict() for result in results], indent=2)


def render_text(results: Iterable[DescriptionResult]) -> str:
    return "\n".join(f"{result.timecode} {result.description}" for result in results)


def render_srt(results: Iterable[DescriptionResult]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        start = format_srt_time(result.start_seconds)
        end = format_srt_time(result.start_seconds + result.duration_seconds)
        blocks.append(f"{index}\n{start} --> {end}\n{result.description}")
    return "\n\n".join(blocks)


def render_results(results: list[DescriptionResult], output_format: str) -> str:
    if output_format == "json":
        return render_json(results)
    if output_format == "srt":
        return render_srt(results)
    if output_format == "text":
        return render_text(results)
    raise ValueError(f"unsupported output format: {output_format}")


def render_gap_results(results: list[GapResult], output_format: str) -> str:
    if output_format == "json":
        return render_gap_json(results)
    if output_format == "srt":
        return render_gap_srt(results)
    if output_format == "text":
        return render_gap_text(results)
    raise ValueError(f"unsupported output format: {output_format}")


def render_compliance_report(report: Any, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report.json_dict(), indent=2)
    if output_format == "text":
        return render_compliance_text(report)
    raise ValueError(f"unsupported output format: {output_format}")


def render_kit(kit: Any, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(kit.json_dict(), indent=2)
    if output_format == "srt":
        return render_kit_srt(kit)
    if output_format == "text":
        return render_kit_text(kit)
    raise ValueError(f"unsupported output format: {output_format}")


def render_edu(kit: Any, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(kit.json_dict(), indent=2)
    if output_format == "srt":
        return render_edu_srt(kit)
    if output_format == "text":
        return render_edu_text(kit)
    raise ValueError(f"unsupported output format: {output_format}")


def render_sports(kit: Any, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(kit.json_dict(), indent=2)
    if output_format == "srt":
        return render_sports_srt(kit)
    if output_format == "text":
        return render_sports_text(kit)
    raise ValueError(f"unsupported output format: {output_format}")


def render_theater(kit: Any, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(kit.json_dict(), indent=2)
    if output_format == "srt":
        return render_theater_srt(kit)
    if output_format == "text":
        return render_theater_text(kit)
    raise ValueError(f"unsupported output format: {output_format}")


def render_podcast(kit: Any, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(kit.json_dict(), indent=2)
    if output_format == "text":
        return render_podcast_text(kit)
    raise ValueError(f"unsupported output format: {output_format}")


def render_ad(kit: Any, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(kit.json_dict(), indent=2)
    if output_format == "srt":
        return render_ad_srt(kit)
    if output_format == "text":
        return render_ad_text(kit)
    raise ValueError(f"unsupported output format: {output_format}")


def render_score(report: Any, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report.json_dict(), indent=2)
    if output_format == "text":
        return render_score_text(report)
    if output_format == "flagged":
        return render_score_text(report, flagged_only=True)
    raise ValueError(f"unsupported output format: {output_format}")


def result_from_api_response(response: dict[str, Any], timestamp: float, duration: float) -> DescriptionResult:
    return DescriptionResult(
        timecode=format_json_time(timestamp),
        description=str(response.get("description") or response.get("narration") or ""),
        objects_detected=_objects_from_response(response),
        latency_ms=_latency_from_response(response),
        start_seconds=timestamp,
        duration_seconds=duration,
    )


def render_gap_json(results: Iterable[GapResult]) -> str:
    return json.dumps([result.json_dict() for result in results], indent=2)


def render_gap_text(results: Iterable[GapResult]) -> str:
    return "\n".join(
        f"{format_gap_time(result.start_sec)} \u2192 {format_gap_time(result.end_sec)} ({format_gap_duration(result.duration_sec)} available)"
        for result in results
    )


def render_gap_srt(results: Iterable[GapResult]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        start = format_srt_time(result.start_sec)
        end = format_srt_time(result.end_sec)
        blocks.append(
            f"{index}\n{start} --> {end}\n[NARRATION GAP \u2014 {format_gap_duration(result.duration_sec)} available]"
        )
    return "\n\n".join(blocks)


def render_compliance_text(report: Any) -> str:
    lines = [
        f"Compliance score: {report.score}",
        f"WCAG level: {report.wcag_level}",
        f"Coverage: {report.coverage_percent:.1f}%",
        f"Max unbroken speech: {format_gap_duration(report.max_unbroken_speech_sec)}",
        "",
        "Criteria:",
    ]
    for key, criterion in report.criteria.items():
        status = "PASS" if criterion.passed else "FAIL"
        level = f"Level {criterion.level}" if criterion.level else "CVAA"
        lines.append(f"- {key} ({level}): {status}")

    lines.extend(["", "Gaps:"])
    if report.gaps:
        lines.extend(
            f"- {format_gap_time(gap.start_sec)} \u2192 {format_gap_time(gap.end_sec)} "
            f"({format_gap_duration(gap.duration_sec)} available, {gap.gap_type})"
            for gap in report.gaps
        )
    else:
        lines.append("- None")

    lines.extend(["", "Recommendations:"])
    if report.recommendations:
        lines.extend(f"- {recommendation}" for recommendation in report.recommendations)
    else:
        lines.append("- No narration gaps found.")

    return "\n".join(lines)


def render_kit_srt(kit: Any) -> str:
    blocks = []
    for narration in kit.narrations:
        start = format_srt_time(narration.start_sec)
        end = format_srt_time(narration.end_sec)
        blocks.append(f"{narration.srt_index}\n{start} --> {end}\n{narration.description}")
    return "\n\n".join(blocks)


def render_kit_text(kit: Any) -> str:
    lines = [
        f"Festival Film Accessibility Kit for {kit.source}",
        f"Duration: {format_gap_duration(kit.duration_seconds)}",
        f"Narration gaps found: {kit.gaps_found}",
        f"Model version: {kit.model_version or 'n/a'}",
        f"Estimated frame cost: ${kit.cost_estimate:.6f}",
        "",
        "Narration script:",
    ]

    if kit.narrations:
        for narration in kit.narrations:
            lines.append(
                f"{narration.srt_index}. {format_gap_time(narration.start_sec)} -> "
                f"{format_gap_time(narration.end_sec)} "
                f"({format_gap_duration(narration.gap_duration_sec)} available, {narration.gap_type})"
            )
            lines.append(narration.description)
            lines.append("")
    else:
        lines.append("No narration gaps found.")
        lines.append("")

    lines.extend(
        [
            "WCAG/CVAA summary:",
            f"Score: {kit.compliance.score}",
            f"WCAG level: {kit.compliance.wcag_level}",
            f"Coverage: {kit.compliance.coverage_percent:.1f}%",
            f"Max unbroken speech: {format_gap_duration(kit.compliance.max_unbroken_speech_sec)}",
        ]
    )
    return "\n".join(lines).rstrip()


def render_edu_srt(kit: Any) -> str:
    blocks = []
    for narration in kit.narrations:
        start = format_srt_time(narration.start_sec)
        end = format_srt_time(narration.end_sec)
        blocks.append(f"{narration.srt_index}\n{start} --> {end}\n{narration.description}")
    return "\n\n".join(blocks)


def render_edu_text(kit: Any) -> str:
    lines = [
        "Educational Video Accessibility Report",
        f"Source: {kit.source} | Duration: {format_gap_duration(kit.duration_seconds)}",
        (
            f"Gaps analyzed: {kit.gaps_analyzed} | Visual moments: {kit.visual_moments} | "
            f"Talking head skipped: {kit.skipped_talking_head}"
        ),
        f"Model version: {kit.model_version or 'n/a'} | Estimated frame cost: ${kit.cost_estimate:.6f}",
        "",
    ]

    if kit.narrations:
        for narration in kit.narrations:
            lines.append(
                f"[{format_gap_time(narration.start_sec)}] "
                f"({format_gap_duration(narration.gap_duration_sec)} available, {narration.gap_type})"
            )
            lines.append(narration.description)
            lines.append("")
    else:
        lines.append("No visually informative frames found in the detected narration gaps.")
        lines.append("")

    lines.extend(
        [
            "WCAG/CVAA Summary",
            (
                f"Score: {kit.compliance.score} | Level: {kit.compliance.wcag_level} | "
                f"Coverage: {kit.compliance.coverage_percent:.1f}% | "
                f"Max unbroken speech: {format_gap_duration(kit.compliance.max_unbroken_speech_sec)}"
            ),
        ]
    )
    return "\n".join(lines).rstrip()


def render_sports_srt(kit: Any) -> str:
    blocks = []
    for index, narration in enumerate(kit.narrations):
        start = narration.timestamp_sec
        if index + 1 < len(kit.narrations):
            end = kit.narrations[index + 1].timestamp_sec
        else:
            end = min(kit.duration_seconds, narration.timestamp_sec + kit.narrate_every_sec)
        if end <= start:
            end = start + max(1.0, min(kit.narrate_every_sec, 4.0))
        blocks.append(
            f"{narration.srt_index}\n"
            f"{format_srt_time(start)} --> {format_srt_time(end)}\n"
            f"{narration.narration}"
        )
    return "\n\n".join(blocks)


def render_sports_text(kit: Any) -> str:
    lines = [
        "Sports Broadcast Describer",
        f"Source: {kit.source} | Duration: {kit.duration_seconds:.1f}s",
        (
            f"Frames analyzed: {kit.frames_analyzed} | Narrations: {kit.narrations_generated} | "
            f"FPS: {kit.fps:.1f} | Narrate every: {kit.narrate_every_sec:.1f}s"
        ),
        (
            f"Rekognition: ${kit.rekognition_cost_estimate:.3f} | "
            f"GPT-4o: ${kit.gpt_cost_estimate:.3f} | Total: ${kit.total_cost_estimate:.3f}"
        ),
        "",
    ]

    if kit.narrations:
        for narration in kit.narrations:
            lines.append(
                f"[{format_gap_time(narration.timestamp_sec)}] "
                f"{len(narration.tracked_objects)} objects tracked ({_sports_object_summary(narration.tracked_objects)})"
            )
            lines.append(narration.narration)
            lines.append("")
    else:
        lines.append("No narrations generated.")
        lines.append("")

    lines.append(f"Model version: {kit.model_version}")
    return "\n".join(lines).rstrip()


def render_theater_srt(kit: Any) -> str:
    blocks = []
    for narration in kit.narrations:
        start = format_srt_time(narration.start_sec)
        end = format_srt_time(narration.end_sec)
        blocks.append(f"{narration.srt_index}\n{start} --> {end}\n{narration.description}")
    return "\n\n".join(blocks)


def render_theater_text(kit: Any) -> str:
    lines = [
        "Theater Mode Accessibility Audio Track",
        (
            f"Source: {kit.source} | Duration: {format_long_duration(kit.duration_seconds)}"
        ),
        (
            f"Gaps processed: {kit.gaps_found} | Audio files: {len(kit.narrations)}"
        ),
        (
            f"Voice: {kit.voice_id} | GPT cost: ${kit.gpt_cost_estimate:.3f} | "
            f"TTS cost: ${kit.tts_cost_estimate:.3f} | Total: ${kit.total_cost_estimate:.3f}"
        ),
        f"Output: {kit.output_dir}",
        "",
    ]

    if kit.narrations:
        for narration in kit.narrations:
            lines.append(
                f"[{format_gap_time(narration.start_sec)}] -> "
                f"[{format_gap_time(narration.end_sec)}] "
                f"({format_gap_duration(narration.gap_duration_sec)} gap, {narration.gap_type})"
            )
            lines.append(narration.description)
            lines.append(f"-> {narration.audio_file}")
            lines.append("")
    else:
        lines.append("No narration gaps found.")
        lines.append("")

    lines.extend(
        [
            "WCAG/CVAA Summary",
            (
                f"Score: {kit.compliance.score} | Level: {kit.compliance.wcag_level} | "
                f"Coverage: {kit.compliance.coverage_percent:.1f}% | "
                f"Max unbroken speech: {format_gap_duration(kit.compliance.max_unbroken_speech_sec)}"
            ),
        ]
    )
    return "\n".join(lines).rstrip()


def render_podcast_text(kit: Any) -> str:
    lines = [
        "Podcast Video Describer",
        f"Source: {kit.source} | Duration: {format_long_duration(kit.duration_seconds)}",
        f"Gaps narrated: {kit.narrations_mixed} | Output: {kit.output_file}",
        (
            f"Voice: {kit.voice_id} | GPT: ${kit.gpt_cost_estimate:.3f} | "
            f"TTS: ${kit.tts_cost_estimate:.3f} | Total: ${kit.total_cost_estimate:.3f}"
        ),
        "",
    ]

    if kit.narrations:
        for narration in kit.narrations:
            lines.append(
                f"[{format_gap_time(narration.start_sec)}] "
                f"({format_gap_duration(narration.gap_duration_sec)} gap, {narration.gap_type})"
            )
            lines.append(narration.description)
            lines.append("")
    else:
        lines.append("No narration gaps found.")
        lines.append("")

    lines.append(f"Work directory: {kit.output_dir}")
    return "\n".join(lines).rstrip()


def render_ad_srt(kit: Any) -> str:
    blocks = []
    for narration in kit.narrations:
        start = format_srt_time(narration.start_sec)
        end = format_srt_time(narration.end_sec)
        blocks.append(f"{narration.srt_index}\n{start} --> {end}\n{narration.description}")
    return "\n\n".join(blocks)


def render_ad_text(kit: Any) -> str:
    lines = [
        "Standard AD Track — WCAG 2.1 Level AA",
        f"Source: {kit.source} | Duration: {format_long_duration(kit.duration_seconds)}",
        f"Gaps processed: {kit.gaps_found} | Voice: {kit.voice_id}",
        (
            f"GPT cost: ${kit.gpt_cost_estimate:.3f} | TTS cost: ${kit.tts_cost_estimate:.3f} | "
            f"Total: ${kit.total_cost_estimate:.3f}"
        ),
        f"Output: {kit.output_dir}",
        "",
    ]

    if kit.narrations:
        for narration in kit.narrations:
            lines.append(
                f"[{format_gap_time(narration.start_sec)}] → "
                f"[{format_gap_time(narration.end_sec)}] "
                f"({format_gap_duration(narration.gap_duration_sec)} gap, {narration.gap_type})"
            )
            lines.append(narration.description)
            lines.append(f"→ {narration.audio_file}")
            lines.append("")
    else:
        lines.append("No narration gaps found.")
        lines.append("")

    lines.extend(
        [
            "WCAG/CVAA Summary",
            (
                f"Score: {kit.compliance.score} | Level: {kit.compliance.wcag_level} | "
                f"Coverage: {kit.compliance.coverage_percent:.1f}% | "
                f"Max unbroken speech: {format_gap_duration(kit.compliance.max_unbroken_speech_sec)}"
            ),
        ]
    )
    return "\n".join(lines).rstrip()


def render_score_text(report: Any, flagged_only: bool = False) -> str:
    visible_scores = [score for score in report.scores if score.flag] if flagged_only else list(report.scores)
    lines = [
        "AD Quality Score Report",
        f"Source: {report.source} | Manifest: {report.manifest}",
        (
            f"Scored: {report.scored} descriptions | Flagged: {report.flagged} | "
            f"Grade: {report.grade} | GPT cost: ${report.gpt_cost_estimate:.3f}"
        ),
        "",
        "Aggregate",
        f"  Accuracy:        {report.aggregate.accuracy:.1f}/10",
        f"  Relevance:       {report.aggregate.relevance:.1f}/10",
        f"  WCAG Compliance: {report.aggregate.wcag_compliance:.1f}/10",
        f"  Conciseness:     {report.aggregate.conciseness:.1f}/10",
        f"  Overall:         {report.aggregate.overall:.1f}/10",
        (
            f"  Within limit:    {report.aggregate.within_limit_pct:.1f}% | "
            f"Present tense: {report.aggregate.tense_ok_pct:.1f}%"
        ),
        "",
    ]

    if visible_scores:
        for score in visible_scores:
            status = "✗ FLAGGED" if score.flag else "✓"
            lines.append(
                f"[{format_gap_time(score.start_sec)}] → [{format_gap_time(score.end_sec)}]  "
                f"overall={_format_brief_score(score.overall)}  words={score.word_count}  {status}"
            )
            lines.append(score.description)
            if score.flag and score.flag_reason:
                lines.append(
                    "  ↳ "
                    f"accuracy={_format_brief_score(score.accuracy)}, "
                    f"relevance={_format_brief_score(score.relevance)}, "
                    f"wcag_compliance={_format_brief_score(score.wcag_compliance)}, "
                    f"conciseness={_format_brief_score(score.conciseness)}"
                    f" — {score.flag_reason}"
                )
            lines.append("")
    else:
        lines.append("No flagged descriptions." if flagged_only else "No descriptions scored.")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_json_time(seconds: float) -> str:
    hours, minutes, secs, millis = _split_time(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_srt_time(seconds: float) -> str:
    hours, minutes, secs, millis = _split_time(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_gap_time(seconds: float) -> str:
    hours, minutes, secs, _millis = _split_time(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_gap_duration(seconds: float) -> str:
    return f"{seconds:.1f}s"


def format_long_duration(seconds: float) -> str:
    hours, minutes, secs, _millis = _split_time(seconds)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _split_time(seconds: float) -> tuple[int, int, int, int]:
    total_ms = max(0, int(round(seconds * 1000)))
    millis = total_ms % 1000
    total_seconds = total_ms // 1000
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return hours, minutes, secs, millis


def _format_brief_score(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _objects_from_response(response: dict[str, Any]) -> list[Any]:
    objects = response.get("objects_detected")
    if isinstance(objects, list):
        return objects
    analysis = response.get("analysis")
    if isinstance(analysis, dict) and isinstance(analysis.get("objects_detected"), list):
        return analysis["objects_detected"]
    return []


def _latency_from_response(response: dict[str, Any]) -> int | None:
    latency = response.get("latency_ms")
    if latency is None:
        performance = response.get("performance")
        if isinstance(performance, dict):
            latency = performance.get("latency_ms") or performance.get("processing_time_ms")
    try:
        return int(float(latency))
    except (TypeError, ValueError):
        return None


def _sports_object_summary(tracked_objects: Iterable[Any]) -> str:
    counts = Counter()
    for tracked_object in tracked_objects:
        label = getattr(tracked_object, "label", None)
        if isinstance(label, str) and label.strip():
            counts[label.strip()] += 1
    if not counts:
        return "none"
    return ", ".join(
        f"{label} x{count}"
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
