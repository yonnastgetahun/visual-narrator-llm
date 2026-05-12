from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .ad import AdResult, assemble_ad_kit
from .characters import CharacterRoster, build_roster_from_audio


EMBEDDINGS_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
REFERENCE_SOURCE = "CMD-AD / AutoAD III (Oxford VGG)"


class BenchmarkError(RuntimeError):
    """Raised when benchmark inputs or external API calls fail."""


@dataclass(frozen=True)
class BenchmarkSample:
    timestamp: str
    vn_timestamp: str
    vn: str
    reference: str
    similarity: float
    word_count_ratio: float
    gap_fit: bool
    overlap_iou: float

    def json_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "vn_timestamp": self.vn_timestamp,
            "vn": self.vn,
            "reference": self.reference,
            "similarity": round(self.similarity, 4),
            "word_count_ratio": round(self.word_count_ratio, 4),
            "gap_fit": self.gap_fit,
            "overlap_iou": round(self.overlap_iou, 4),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    clip: str
    reference_source: str
    gaps_matched: int
    semantic_similarity_mean: float
    semantic_similarity_median: float
    word_count_ratio_mean: float
    gap_fit_rate: float
    quality_ratio: str
    vn_samples: list[BenchmarkSample]

    def json_dict(self) -> dict[str, Any]:
        return {
            "clip": self.clip,
            "reference_source": self.reference_source,
            "gaps_matched": self.gaps_matched,
            "semantic_similarity_mean": round(self.semantic_similarity_mean, 4),
            "semantic_similarity_median": round(self.semantic_similarity_median, 4),
            "word_count_ratio_mean": round(self.word_count_ratio_mean, 4),
            "gap_fit_rate": round(self.gap_fit_rate, 4),
            "quality_ratio": self.quality_ratio,
            "vn_samples": [sample.json_dict() for sample in self.vn_samples],
        }


@dataclass(frozen=True)
class _Cue:
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class _MatchedCue:
    reference: _Cue
    narration: Any
    overlap_iou: float


def benchmark_video(
    video_path: Path,
    reference_path: Path,
    output_path: Path,
    min_gap: float = 2.0,
    voice_id: str | None = None,
    character_sheet: Path | None = None,
    auto_detect_names: bool = True,
) -> BenchmarkReport:
    resolved_video_path = video_path.expanduser().resolve()
    resolved_reference_path = reference_path.expanduser().resolve()
    resolved_output_path = output_path.expanduser().resolve()

    if not resolved_video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not resolved_reference_path.exists():
        raise FileNotFoundError(f"reference not found: {reference_path}")

    # Build character roster — sheet takes priority, audio detection fills gaps
    roster = CharacterRoster()
    if character_sheet and character_sheet.exists():
        roster = roster.merge(CharacterRoster.from_file(character_sheet))
    if auto_detect_names and roster.is_empty():
        roster = roster.merge(build_roster_from_audio(resolved_video_path))
    character_context = roster.to_prompt_fragment()

    reference_cues = _load_srt(resolved_reference_path)
    with tempfile.TemporaryDirectory(prefix="vn-benchmark-") as tmp:
        pipeline_output_dir = Path(tmp) / "ad-output"
        ad_result = assemble_ad_kit(
            resolved_video_path,
            min_gap=min_gap,
            voice_id=voice_id,
            source_label=resolved_video_path.name,
            output_dir=pipeline_output_dir,
            character_context=character_context,
        )

    matched = _match_reference_cues(reference_cues, ad_result)
    similarities = _semantic_similarities(matched)
    samples = _build_samples(matched, similarities)

    semantic_scores = [sample.similarity for sample in samples]
    word_ratios = [sample.word_count_ratio for sample in samples]
    gap_fit_values = [1.0 if sample.gap_fit else 0.0 for sample in samples]

    mean_similarity = statistics.fmean(semantic_scores) if semantic_scores else 0.0
    median_similarity = statistics.median(semantic_scores) if semantic_scores else 0.0
    mean_word_ratio = statistics.fmean(word_ratios) if word_ratios else 0.0
    gap_fit_rate = statistics.fmean(gap_fit_values) if gap_fit_values else 0.0

    report = BenchmarkReport(
        clip=resolved_video_path.name,
        reference_source=REFERENCE_SOURCE,
        gaps_matched=len(samples),
        semantic_similarity_mean=mean_similarity,
        semantic_similarity_median=median_similarity,
        word_count_ratio_mean=mean_word_ratio,
        gap_fit_rate=gap_fit_rate,
        quality_ratio=_format_percentage(mean_similarity),
        vn_samples=samples,
    )

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(report.json_dict(), indent=2), encoding="utf-8")
    return report


def _load_srt(path: Path) -> list[_Cue]:
    raw_text = path.read_text(encoding="utf-8-sig")
    blocks = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    cues: list[_Cue] = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise BenchmarkError(f"Invalid SRT block in {path}: {block!r}")
        time_line = lines[1]
        if "-->" not in time_line:
            raise BenchmarkError(f"Invalid SRT timing line in {path}: {time_line!r}")
        start_raw, end_raw = [part.strip() for part in time_line.split("-->", maxsplit=1)]
        cues.append(
            _Cue(
                start_sec=_parse_srt_time(start_raw),
                end_sec=_parse_srt_time(end_raw),
                text=" ".join(lines[2:]).strip(),
            )
        )
    return cues


def _parse_srt_time(value: str) -> float:
    try:
        hours_part, minutes_part, seconds_part = value.split(":")
        seconds_text, millis_text = seconds_part.split(",")
        hours = int(hours_part)
        minutes = int(minutes_part)
        seconds = int(seconds_text)
        millis = int(millis_text)
    except ValueError as exc:
        raise BenchmarkError(f"Invalid SRT timestamp: {value!r}") from exc
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _match_reference_cues(reference_cues: list[_Cue], ad_result: AdResult) -> list[_MatchedCue]:
    matches: list[_MatchedCue] = []
    for cue in reference_cues:
        best_match = None
        best_iou = 0.0
        for narration in ad_result.narrations:
            overlap_iou = _window_iou(cue.start_sec, cue.end_sec, narration.start_sec, narration.end_sec)
            if overlap_iou > best_iou:
                best_iou = overlap_iou
                best_match = narration
        if best_match is not None and best_iou > 0:
            matches.append(_MatchedCue(reference=cue, narration=best_match, overlap_iou=best_iou))
    return matches


def _window_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return intersection / union


def _semantic_similarities(matches: list[_MatchedCue]) -> list[float]:
    if not matches:
        return []

    unique_texts = list(
        dict.fromkeys(
            [match.reference.text for match in matches]
            + [match.narration.description for match in matches]
        )
    )
    embeddings = _fetch_embeddings(unique_texts)
    embedding_map = dict(zip(unique_texts, embeddings, strict=True))
    return [
        _cosine_similarity(
            embedding_map[match.reference.text],
            embedding_map[match.narration.description],
        )
        for match in matches
    ]


def _fetch_embeddings(texts: list[str]) -> list[list[float]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise BenchmarkError("OPENAI_API_KEY is not set.")

    payload = {
        "model": EMBEDDINGS_MODEL,
        "input": texts,
    }

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.post(
                OPENAI_EMBEDDINGS_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise BenchmarkError(f"OpenAI embeddings API error {exc.response.status_code}: {exc.response.text}") from exc
    except httpx.RequestError as exc:
        raise BenchmarkError(f"OpenAI embeddings request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise BenchmarkError("OpenAI embeddings API returned invalid JSON.") from exc

    rows = data.get("data")
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise BenchmarkError("OpenAI embeddings API returned an unexpected payload.")

    embeddings: list[list[float]] = []
    for row in rows:
        embedding = row.get("embedding") if isinstance(row, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise BenchmarkError("OpenAI embeddings API returned an invalid embedding vector.")
        embeddings.append([float(value) for value in embedding])
    return embeddings


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _build_samples(matches: list[_MatchedCue], similarities: list[float]) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    for match, similarity in zip(matches, similarities, strict=True):
        narration = match.narration
        audio_fit = narration.audio_fit if isinstance(narration.audio_fit, dict) else {}
        samples.append(
            BenchmarkSample(
                timestamp=_format_time_window(match.reference.start_sec, match.reference.end_sec),
                vn_timestamp=_format_time_window(narration.start_sec, narration.end_sec),
                vn=narration.description,
                reference=match.reference.text,
                similarity=similarity,
                word_count_ratio=_word_count_ratio(narration.description, match.reference.text),
                gap_fit=bool(audio_fit.get("fit")),
                overlap_iou=match.overlap_iou,
            )
        )
    return samples


def _word_count_ratio(left: str, right: str) -> float:
    left_words = len(left.split())
    right_words = len(right.split())
    if left_words == 0 or right_words == 0:
        return 0.0
    return min(left_words, right_words) / max(left_words, right_words)


def _format_time_window(start_sec: float, end_sec: float) -> str:
    return f"{_format_srt_time(start_sec)} --> {_format_srt_time(end_sec)}"


def _format_srt_time(seconds: float) -> str:
    total_millis = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"
