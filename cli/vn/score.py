from __future__ import annotations

import json
import mimetypes
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .api import encode_file_base64
from .frame import FrameExtractionError, extract_frames_at


SCORE_PROMPT_TEMPLATE = """You are an expert audio description quality reviewer. You will be shown a video frame and an audio description that was written to describe it.

Score the description on each dimension from 0 to 10:

- accuracy: Does the description correctly describe what is visually present in the frame?
- relevance: Does it focus on information that is important for understanding the content (not background clutter)?
- wcag_compliance: Is it factual, objective, present tense, and free of emotional interpretation?
- conciseness: Is it appropriately brief without omitting critical information?

Also check:
- word_count: Count the words in the description
- tense_ok: true if the description uses present tense throughout
- within_limit: true if word_count <= {word_limit}

Respond with JSON only, no explanation:
{{
  "accuracy": <0-10>,
  "relevance": <0-10>,
  "wcag_compliance": <0-10>,
  "conciseness": <0-10>,
  "overall": <0-10>,
  "word_count": <int>,
  "tense_ok": <bool>,
  "within_limit": <bool>,
  "flag": <bool - true if any dimension < 6 or within_limit is false>,
  "flag_reason": "<short explanation if flag is true, else null>"
}}

Audio description:
{description}
"""

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o"
SCORE_COST_PER_FRAME = 0.0013

GRADE_THRESHOLDS = [
    (9.0, "A"),
    (8.0, "B+"),
    (7.0, "B"),
    (6.0, "C"),
    (0.0, "F"),
]


class ScoreError(RuntimeError):
    """Raised when GPT-4o scoring fails or returns invalid JSON."""


@dataclass(frozen=True)
class DescriptionScore:
    srt_index: int
    start_sec: float
    end_sec: float
    frame_timestamp_sec: float
    description: str
    accuracy: float
    relevance: float
    wcag_compliance: float
    conciseness: float
    overall: float
    word_count: int
    tense_ok: bool
    within_limit: bool
    flag: bool
    flag_reason: str | None
    gpt_cost: float

    def json_dict(self) -> dict[str, Any]:
        return {
            "srt_index": self.srt_index,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "frame_timestamp_sec": round(self.frame_timestamp_sec, 3),
            "description": self.description,
            "accuracy": round(self.accuracy, 3),
            "relevance": round(self.relevance, 3),
            "wcag_compliance": round(self.wcag_compliance, 3),
            "conciseness": round(self.conciseness, 3),
            "overall": round(self.overall, 3),
            "word_count": self.word_count,
            "tense_ok": self.tense_ok,
            "within_limit": self.within_limit,
            "flag": self.flag,
            "flag_reason": self.flag_reason,
            "gpt_cost": round(self.gpt_cost, 6),
        }


@dataclass(frozen=True)
class ScoreAggregate:
    accuracy: float
    relevance: float
    wcag_compliance: float
    conciseness: float
    overall: float
    within_limit_pct: float
    tense_ok_pct: float

    def json_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 3),
            "relevance": round(self.relevance, 3),
            "wcag_compliance": round(self.wcag_compliance, 3),
            "conciseness": round(self.conciseness, 3),
            "overall": round(self.overall, 3),
            "within_limit_pct": round(self.within_limit_pct, 3),
            "tense_ok_pct": round(self.tense_ok_pct, 3),
        }


@dataclass(frozen=True)
class ScoreReport:
    source: str
    manifest: str
    scored: int
    flagged: int
    word_limit: int
    aggregate: ScoreAggregate
    grade: str
    gpt_cost_estimate: float
    scores: list[DescriptionScore]

    def json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "manifest": self.manifest,
            "scored": self.scored,
            "flagged": self.flagged,
            "word_limit": self.word_limit,
            "aggregate": self.aggregate.json_dict(),
            "grade": self.grade,
            "gpt_cost_estimate": round(self.gpt_cost_estimate, 6),
            "scores": [score.json_dict() for score in self.scores],
        }


@dataclass(frozen=True)
class _ManifestNarration:
    srt_index: int
    start_sec: float
    end_sec: float
    frame_timestamp_sec: float
    description: str


def score_manifest(
    manifest_path: Path,
    video_source: Path,
    word_limit: int | None = None,
    min_score: float = 6.0,
    output_dir: Path | None = None,
    source_label: str | None = None,
    manifest_label: str | None = None,
) -> ScoreReport:
    if min_score < 0 or min_score > 10:
        raise ValueError("--min-score must be between 0 and 10")
    if word_limit is not None and word_limit <= 0:
        raise ValueError("--word-limit must be greater than 0")

    resolved_manifest_path = manifest_path.expanduser()
    if not resolved_manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest_data = _load_manifest(resolved_manifest_path)
    narrations = _load_narrations(manifest_data)
    resolved_word_limit = word_limit or _detect_word_limit(manifest_data)
    requested_output_dir = output_dir or Path("./vn-score-output")
    resolved_output_dir = requested_output_dir.expanduser()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    scores: list[DescriptionScore] = []
    with tempfile.TemporaryDirectory(prefix="vn-score-frames-") as tmp:
        frame_root = Path(tmp)
        for narration in narrations:
            try:
                frames = extract_frames_at(
                    video_source,
                    [narration.frame_timestamp_sec],
                    frame_root / f"{narration.srt_index:05d}",
                )
            except FrameExtractionError as exc:
                print(
                    (
                        f"Warning: skipping narration {narration.srt_index} at "
                        f"{narration.frame_timestamp_sec:.3f}s: {exc}"
                    ),
                    file=sys.stderr,
                )
                continue
            score = _score_description(
                frames[0].path,
                narration,
                word_limit=resolved_word_limit,
                min_score=min_score,
            )
            scores.append(score)

    aggregate = _aggregate_scores(scores)
    flagged = sum(1 for score in scores if score.flag)
    report = ScoreReport(
        source=source_label or str(video_source),
        manifest=manifest_label or str(manifest_path),
        scored=len(scores),
        flagged=flagged,
        word_limit=resolved_word_limit,
        aggregate=aggregate,
        grade=_grade_for_score(aggregate.overall),
        gpt_cost_estimate=sum(score.gpt_cost for score in scores),
        scores=scores,
    )

    report_path = resolved_output_dir / "score-report.json"
    report_path.write_text(json.dumps(report.json_dict(), indent=2), encoding="utf-8")
    return report


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"manifest.json is not valid JSON: {manifest_path}") from exc
    if not isinstance(data, dict):
        raise ValueError("manifest.json must contain a JSON object")
    return data


def _load_narrations(manifest_data: dict[str, Any]) -> list[_ManifestNarration]:
    narrations_data = manifest_data.get("narrations")
    if narrations_data is None:
        raise ValueError("manifest.json has no narrations field")
    if not isinstance(narrations_data, list):
        raise ValueError("manifest.json narrations field must be a list")

    narrations: list[_ManifestNarration] = []
    for index, item in enumerate(narrations_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"manifest narration {index} must be an object")
        start_sec = _coerce_float(item.get("start_sec"), f"narration {index} start_sec")
        end_sec = _coerce_float(item.get("end_sec"), f"narration {index} end_sec")
        frame_timestamp_raw = item.get("frame_timestamp_sec")
        frame_timestamp_sec = (
            _coerce_float(frame_timestamp_raw, f"narration {index} frame_timestamp_sec")
            if frame_timestamp_raw is not None
            else (start_sec + end_sec) / 2
        )
        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"manifest narration {index} description must be a non-empty string")
        srt_index_raw = item.get("srt_index", index)
        if isinstance(srt_index_raw, bool):
            raise ValueError(f"manifest narration {index} srt_index must be an integer")
        try:
            srt_index = int(srt_index_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"manifest narration {index} srt_index must be an integer") from exc
        narrations.append(
            _ManifestNarration(
                srt_index=srt_index,
                start_sec=start_sec,
                end_sec=end_sec,
                frame_timestamp_sec=frame_timestamp_sec,
                description=description.strip(),
            )
        )
    return narrations


def _detect_word_limit(manifest_data: dict[str, Any]) -> int:
    explicit_limit = manifest_data.get("word_limit")
    if explicit_limit is not None:
        try:
            parsed_limit = int(explicit_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("manifest word_limit must be an integer") from exc
        if parsed_limit <= 0:
            raise ValueError("manifest word_limit must be greater than 0")
        return parsed_limit
    if "compliance_level" in manifest_data:
        return 30
    return 60


def _score_description(
    frame_path: Path,
    narration: _ManifestNarration,
    word_limit: int,
    min_score: float,
) -> DescriptionScore:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ScoreError("OPENAI_API_KEY is not set.")

    mime_type = mimetypes.guess_type(frame_path.name)[0] or "image/jpeg"
    prompt = SCORE_PROMPT_TEMPLATE.format(
        word_limit=word_limit,
        description=narration.description,
    )
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 250,
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
        raise ScoreError(f"OpenAI API error {exc.response.status_code}: {exc.response.text}") from exc
    except httpx.RequestError as exc:
        raise ScoreError(f"OpenAI request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise ScoreError(f"OpenAI returned invalid JSON: {response.text[:300]}") from exc
    if not isinstance(data, dict):
        raise ScoreError("OpenAI returned a non-object response.")

    raw_content = _assistant_text_from_response(data).strip()
    score_payload = _parse_score_payload(raw_content)
    accuracy = _bounded_score(score_payload.get("accuracy"), "accuracy")
    relevance = _bounded_score(score_payload.get("relevance"), "relevance")
    wcag_compliance = _bounded_score(score_payload.get("wcag_compliance"), "wcag_compliance")
    conciseness = _bounded_score(score_payload.get("conciseness"), "conciseness")
    overall = _bounded_score(score_payload.get("overall"), "overall")
    word_count = _coerce_int(score_payload.get("word_count"), "word_count")
    tense_ok = _coerce_bool(score_payload.get("tense_ok"), "tense_ok")
    within_limit = _coerce_bool(score_payload.get("within_limit"), "within_limit")

    dimension_scores = {
        "accuracy": accuracy,
        "relevance": relevance,
        "wcag_compliance": wcag_compliance,
        "conciseness": conciseness,
        "overall": overall,
    }
    flag = any(value < min_score for value in dimension_scores.values()) or not within_limit
    raw_flag_reason = score_payload.get("flag_reason")
    flag_reason = _build_flag_reason(
        dimension_scores,
        within_limit,
        min_score,
        raw_flag_reason if isinstance(raw_flag_reason, str) else None,
    )

    return DescriptionScore(
        srt_index=narration.srt_index,
        start_sec=narration.start_sec,
        end_sec=narration.end_sec,
        frame_timestamp_sec=narration.frame_timestamp_sec,
        description=narration.description,
        accuracy=accuracy,
        relevance=relevance,
        wcag_compliance=wcag_compliance,
        conciseness=conciseness,
        overall=overall,
        word_count=word_count,
        tense_ok=tense_ok,
        within_limit=within_limit,
        flag=flag,
        flag_reason=flag_reason if flag else None,
        gpt_cost=SCORE_COST_PER_FRAME,
    )


def _assistant_text_from_response(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ScoreError("OpenAI response did not include choices.")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ScoreError("OpenAI response did not include a valid message.")

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
    raise ScoreError("OpenAI response content was not text.")


def _parse_score_payload(raw_content: str) -> dict[str, Any]:
    normalized = raw_content.strip()
    if normalized.startswith("```"):
        normalized = normalized.strip("`")
        if normalized.startswith("json"):
            normalized = normalized[4:].strip()
    try:
        data = json.loads(normalized)
    except ValueError as exc:
        raise ScoreError(f"GPT-4o returned non-JSON scoring output: {raw_content}") from exc
    if not isinstance(data, dict):
        raise ScoreError(f"GPT-4o returned non-object scoring output: {raw_content}")
    return data


def _aggregate_scores(scores: list[DescriptionScore]) -> ScoreAggregate:
    if not scores:
        return ScoreAggregate(
            accuracy=0.0,
            relevance=0.0,
            wcag_compliance=0.0,
            conciseness=0.0,
            overall=0.0,
            within_limit_pct=0.0,
            tense_ok_pct=0.0,
        )

    total = float(len(scores))
    return ScoreAggregate(
        accuracy=sum(score.accuracy for score in scores) / total,
        relevance=sum(score.relevance for score in scores) / total,
        wcag_compliance=sum(score.wcag_compliance for score in scores) / total,
        conciseness=sum(score.conciseness for score in scores) / total,
        overall=sum(score.overall for score in scores) / total,
        within_limit_pct=100.0 * sum(1 for score in scores if score.within_limit) / total,
        tense_ok_pct=100.0 * sum(1 for score in scores if score.tense_ok) / total,
    )


def _grade_for_score(overall: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if overall >= threshold:
            return grade
    return "F"


def _coerce_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ScoreError(f"GPT-4o returned invalid {field_name}: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ScoreError(f"GPT-4o returned invalid {field_name}: {value!r}") from exc


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ScoreError(f"GPT-4o returned invalid {field_name}: {value!r}")


def _bounded_score(value: Any, field_name: str) -> float:
    try:
        score = _coerce_float(value, field_name)
    except ValueError as exc:
        raise ScoreError(f"GPT-4o returned invalid {field_name}: {value!r}") from exc
    if score < 0 or score > 10:
        raise ScoreError(f"GPT-4o returned {field_name} outside 0-10: {score!r}")
    return score


def _build_flag_reason(
    dimension_scores: dict[str, float],
    within_limit: bool,
    min_score: float,
    raw_flag_reason: str | None,
) -> str | None:
    reasons: list[str] = []
    low_dimensions = [name for name, value in dimension_scores.items() if value < min_score]
    if low_dimensions:
        reasons.append(f"below threshold on {', '.join(low_dimensions)} (< {_format_score(min_score)})")
    if not within_limit:
        reasons.append("exceeds word limit")
    if raw_flag_reason:
        normalized = raw_flag_reason.strip()
        if normalized and normalized not in reasons:
            reasons.append(normalized)
    if not reasons:
        return None
    return " - ".join(reasons)


def _format_score(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")
