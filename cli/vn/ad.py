from __future__ import annotations

import json
import logging
import mimetypes
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import httpx

from .api import encode_file_base64
from .compliance import ComplianceReport, analyze_compliance
from .frame import extract_frames_at
from .gaps import detect_gaps
from .output import GapResult, render_ad_srt


LOGGER = logging.getLogger(__name__)

AD_PROMPT_TEMPLATE = (
    "You are writing a professional audio description for accessibility compliance (WCAG 2.1 Level AA).\n"
    "Describe the key visual information concisely and objectively. Focus on:\n"
    "- Actions and movements relevant to understanding the content\n"
    "- Text that appears on screen\n"
    "- Changes in scene or setting\n"
    "- Expressions or gestures that convey meaning not audible in the dialogue\n"
    "Do not attempt to identify individuals by name or face. "
    "Describe visible actions, body positions, clothing, and movements only. "
    "If character names are provided in context, use them — otherwise refer to people by appearance or role.\n"
    "Do not describe irrelevant background details. Do not interpret emotions unless clearly expressed.\n"
    "Use present tense. One to two short sentences. "
    "You have {gap_seconds:.1f} seconds of silence to fill. Maximum {max_words} words. "
    "Be clear and factual."
)

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
AD_DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o"
QWEN_MODEL = "qwen2.5-vl"
HF_INFERENCE_URL = "https://router.huggingface.co/ovhcloud/v1/chat/completions"
HF_MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
HF_PROVIDER_MODEL_ID = "Qwen2.5-VL-72B-Instruct"
SUPPORTED_VISION_MODELS = {OPENAI_MODEL, "gpt-4.1-mini", QWEN_MODEL}
AD_COST_PER_FRAME = 0.0013
AD_TTS_COST_PER_CHAR = 0.0003
WORDS_PER_SECOND = 2.5
MIN_AD_WORDS = 5
MAX_AD_WORDS = 30
SHORTER_RETRY_FACTOR = 0.7
MAX_AUDIO_FIT_RETRIES = 2
OVERRUN_BUFFER = 0.15
TRUNCATION_STEP_SEC = 0.05
MAX_TRUNCATION_ATTEMPTS = 3


class AdDescriptionError(RuntimeError):
    """Raised when GPT-4o frame description fails."""


class AdTTSError(RuntimeError):
    """Raised when ElevenLabs speech synthesis fails."""


@dataclass(frozen=True)
class AudioGapFitMetrics:
    gap_sec: float
    audio_sec: float
    fit: bool
    retries: int
    truncated: bool
    fit_ratio: float
    overrun_attempts: int
    word_limit: int
    max_allowed_sec: float
    attempt_audio_sec: tuple[float, ...] = field(default_factory=tuple)
    attempt_word_limits: tuple[int, ...] = field(default_factory=tuple)

    def json_dict(self) -> dict[str, Any]:
        return {
            "gap_sec": round(self.gap_sec, 3),
            "audio_sec": round(self.audio_sec, 3),
            "fit": self.fit,
            "retries": self.retries,
            "truncated": self.truncated,
            "fit_ratio": round(self.fit_ratio, 3),
            "overrun_attempts": self.overrun_attempts,
            "word_limit": self.word_limit,
            "max_allowed_sec": round(self.max_allowed_sec, 3),
            "attempt_audio_sec": [round(value, 3) for value in self.attempt_audio_sec],
            "attempt_word_limits": list(self.attempt_word_limits),
        }


@dataclass(frozen=True)
class GeneratedAdNarration:
    description: str
    model_version: str
    audio_bytes: bytes = field(repr=False)
    character_count: int = 0
    audio_duration_sec: float = 0.0
    audio_fit: AudioGapFitMetrics = field(
        default_factory=lambda: AudioGapFitMetrics(
            gap_sec=0.0,
            audio_sec=0.0,
            fit=True,
            retries=0,
            truncated=False,
            fit_ratio=0.0,
            overrun_attempts=0,
            word_limit=0,
            max_allowed_sec=0.0,
        )
    )


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
    audio_duration_sec: float
    audio_fit: dict[str, Any]
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
            "audio_duration_sec": round(self.audio_duration_sec, 3),
            "audio_fit": self.audio_fit,
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
    character_context: str | None = None,
) -> AdResult:
    requested_output_dir = output_dir or Path("./vn-ad-output")
    resolved_output_dir = requested_output_dir.expanduser()
    audio_dir = resolved_output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    gaps = detect_gaps(source, min_gap=min_gap)
    narrations: list[AdNarrationEntry] = []
    audio_gap_fit_metrics: list[AudioGapFitMetrics] = []
    model_versions: list[str] = []
    resolved_voice_id = (
        voice_id
        or os.getenv("ELEVENLABS_VOICE_ADAM")
        or AD_DEFAULT_VOICE_ID
    )

    if gaps:
        with tempfile.TemporaryDirectory(prefix="vn-ad-frames-") as tmp:
            frame_dir = Path(tmp)
            frames_per_gap = 3 if _vision_model() == QWEN_MODEL else 1
            all_timestamps: list[float] = []
            for gap in gaps:
                all_timestamps.extend(_gap_timestamps(gap, frames_per_gap))
            all_frames = extract_frames_at(source, all_timestamps, frame_dir)
            frame_groups = [
                all_frames[i * frames_per_gap:(i + 1) * frames_per_gap]
                for i in range(len(gaps))
            ]
            scene_context: list[str] = []
            for index, (gap, frame_group) in enumerate(zip(gaps, frame_groups, strict=True), start=1):
                prefix_parts = []
                if character_context:
                    prefix_parts.append(character_context)
                if scene_context:
                    prefix_parts.append("Recent descriptions:\n" + "\n".join(scene_context[-5:]))
                prefix = "\n\n".join(prefix_parts) or None
                narration = generate_gap_aware_ad_narration(
                    [f.path for f in frame_group],
                    gap.duration_sec,
                    resolved_voice_id,
                    system_prompt_prefix=prefix,
                )
                if narration.description:
                    scene_context.append(narration.description)
                if narration.model_version:
                    model_versions.append(narration.model_version)
                audio_filename = f"{index:05d}_{int(gap.start_sec * 1000):07d}.mp3"
                audio_relative_path = Path("audio") / audio_filename
                _write_audio_file(resolved_output_dir / audio_relative_path, narration.audio_bytes)
                audio_gap_fit_metrics.append(narration.audio_fit)
                narrations.append(
                    AdNarrationEntry(
                        srt_index=index,
                        start_sec=gap.start_sec,
                        end_sec=gap.end_sec,
                        gap_duration_sec=gap.duration_sec,
                        gap_type=gap.gap_type,
                        frame_timestamp_sec=frame_group[0].timestamp,
                        description=narration.description,
                        audio_file=audio_relative_path.as_posix(),
                        audio_duration_sec=narration.audio_duration_sec,
                        audio_fit=narration.audio_fit.json_dict(),
                        gpt_cost=AD_COST_PER_FRAME,
                        tts_cost=narration.character_count * AD_TTS_COST_PER_CHAR,
                    )
                )

    compliance = analyze_compliance(
        source,
        min_gap=min_gap,
        gaps=gaps,
        audio_gap_fit=build_audio_gap_fit_summary(audio_gap_fit_metrics),
    )
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


def target_words(gap_duration_sec: float) -> int:
    if gap_duration_sec <= 0:
        return MIN_AD_WORDS
    return max(MIN_AD_WORDS, min(MAX_AD_WORDS, int(gap_duration_sec * WORDS_PER_SECOND)))


def fits_in_gap(audio_sec: float, gap_sec: float) -> bool:
    return audio_sec <= max(0.0, gap_sec - OVERRUN_BUFFER)


def audio_duration_sec(audio_bytes: bytes) -> float:
    if shutil.which("ffprobe") is None:
        raise AdTTSError("ffprobe is required to verify narration duration. Install ffmpeg first.")

    with tempfile.NamedTemporaryFile(prefix="vn-audio-fit-", suffix=".mp3") as handle:
        handle.write(audio_bytes)
        handle.flush()
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                handle.name,
            ],
            capture_output=True,
            check=False,
        )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
        raise AdTTSError(f"ffprobe failed while measuring narration audio: {stderr or 'unknown error'}")

    try:
        payload = json.loads(completed.stdout.decode("utf-8") or "{}")
    except ValueError as exc:
        raise AdTTSError("ffprobe returned invalid JSON while measuring narration audio.") from exc

    format_duration = ((payload.get("format") or {}).get("duration"))
    try:
        if format_duration is not None:
            return float(format_duration)
    except (TypeError, ValueError):
        pass

    for stream in payload.get("streams", []):
        duration = stream.get("duration")
        try:
            if duration is not None:
                return float(duration)
        except (TypeError, ValueError):
            continue
    raise AdTTSError("ffprobe did not report narration audio duration.")


def truncate_audio(audio_bytes: bytes, max_sec: float) -> bytes:
    if shutil.which("ffmpeg") is None:
        raise AdTTSError("ffmpeg is required to truncate narration audio. Install ffmpeg first.")

    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            "pipe:0",
            "-t",
            f"{max_sec:.3f}",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
        raise AdTTSError(f"ffmpeg failed while truncating narration audio: {stderr or 'unknown error'}")
    if not completed.stdout:
        raise AdTTSError("ffmpeg returned empty audio while truncating narration.")
    return completed.stdout


def synthesize_speech_bytes(text: str, voice_id: str) -> tuple[bytes, int]:
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


def generate_gap_aware_ad_narration(
    frame_paths: Sequence[Path],
    gap_duration_sec: float,
    voice_id: str,
    system_prompt_prefix: str | None = None,
) -> GeneratedAdNarration:
    attempt_audio_sec: list[float] = []
    attempt_word_limits: list[int] = []
    model_versions: list[str] = []
    base_word_limit = target_words(gap_duration_sec)
    last_audio_bytes = b""
    last_description = ""
    last_character_count = 0
    last_duration = 0.0
    last_word_limit = base_word_limit

    for retry in range(MAX_AUDIO_FIT_RETRIES + 1):
        word_limit = _retry_word_limit(base_word_limit, retry)
        description, model_version = _describe_frame_for_ad(
            frame_paths,
            gap_duration_sec,
            max_words=word_limit,
            system_prompt_prefix=system_prompt_prefix,
        )
        if model_version:
            model_versions.append(model_version)
        audio_bytes, character_count = synthesize_speech_bytes(description, voice_id)
        duration = audio_duration_sec(audio_bytes)

        attempt_audio_sec.append(duration)
        attempt_word_limits.append(word_limit)
        last_audio_bytes = audio_bytes
        last_description = description
        last_character_count = character_count
        last_duration = duration
        last_word_limit = word_limit

        if fits_in_gap(duration, gap_duration_sec):
            metrics = AudioGapFitMetrics(
                gap_sec=gap_duration_sec,
                audio_sec=duration,
                fit=True,
                retries=retry,
                truncated=False,
                fit_ratio=_fit_ratio(duration, gap_duration_sec),
                overrun_attempts=retry,
                word_limit=word_limit,
                max_allowed_sec=_max_allowed_audio_sec(gap_duration_sec),
                attempt_audio_sec=tuple(attempt_audio_sec),
                attempt_word_limits=tuple(attempt_word_limits),
            )
            _log_audio_gap_fit(metrics)
            return GeneratedAdNarration(
                description=description,
                model_version=_combine_model_versions(model_versions),
                audio_bytes=audio_bytes,
                character_count=character_count,
                audio_duration_sec=duration,
                audio_fit=metrics,
            )

    truncated_audio, truncated_duration = _truncate_to_fit(last_audio_bytes, gap_duration_sec)
    metrics = AudioGapFitMetrics(
        gap_sec=gap_duration_sec,
        audio_sec=truncated_duration,
        fit=True,
        retries=MAX_AUDIO_FIT_RETRIES,
        truncated=True,
        fit_ratio=_fit_ratio(truncated_duration, gap_duration_sec),
        overrun_attempts=len(attempt_audio_sec),
        word_limit=last_word_limit,
        max_allowed_sec=_max_allowed_audio_sec(gap_duration_sec),
        attempt_audio_sec=tuple([*attempt_audio_sec, truncated_duration]),
        attempt_word_limits=tuple(attempt_word_limits),
    )
    _log_audio_gap_fit(metrics)
    return GeneratedAdNarration(
        description=last_description,
        model_version=_combine_model_versions(model_versions),
        audio_bytes=truncated_audio,
        character_count=last_character_count,
        audio_duration_sec=truncated_duration,
        audio_fit=metrics,
    )


def build_audio_gap_fit_summary(metrics: Sequence[AudioGapFitMetrics]) -> dict[str, Any]:
    total = len(metrics)
    if total == 0:
        return {
            "narrations_evaluated": 0,
            "overrun_count": 0,
            "overrun_rate": 0.0,
            "retry_count": 0,
            "truncation_count": 0,
            "average_fit_ratio": 0.0,
            "buffer_sec": OVERRUN_BUFFER,
        }

    overrun_count = sum(1 for item in metrics if item.overrun_attempts > 0)
    retry_count = sum(item.retries for item in metrics)
    truncation_count = sum(1 for item in metrics if item.truncated)
    average_fit_ratio = sum(item.fit_ratio for item in metrics) / total
    return {
        "narrations_evaluated": total,
        "overrun_count": overrun_count,
        "overrun_rate": round(overrun_count / total, 3),
        "retry_count": retry_count,
        "truncation_count": truncation_count,
        "average_fit_ratio": round(average_fit_ratio, 3),
        "buffer_sec": OVERRUN_BUFFER,
    }


def _describe_frame_hf(
    frame_paths: Sequence[Path],
    gap_duration_sec: float,
    max_words: int | None = None,
    system_prompt_prefix: str | None = None,
) -> tuple[str, str]:
    api_key = os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_VULCAN26") or os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise AdDescriptionError("HF_TOKEN is not set. Required for Qwen2.5-VL via HuggingFace.")
    if not frame_paths:
        raise AdDescriptionError("At least one frame is required for AD generation.")

    resolved_word_limit = max_words or target_words(gap_duration_sec)
    prompt = "\n\n".join(
        part
        for part in (
            _frame_prompt_prefix(len(frame_paths)),
            AD_PROMPT_TEMPLATE.format(gap_seconds=gap_duration_sec, max_words=resolved_word_limit),
        )
        if part
    )
    content: list[dict[str, Any]] = []
    for frame_path in frame_paths:
        mime_type = mimetypes.guess_type(frame_path.name)[0] or "image/jpeg"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encode_file_base64(frame_path)}"},
        })
    content.append({"type": "text", "text": prompt})

    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if system_prompt_prefix:
        messages.insert(0, {"role": "system", "content": system_prompt_prefix})

    payload = {"model": HF_PROVIDER_MODEL_ID, "messages": messages, "max_tokens": 150}

    try:
        with httpx.Client(timeout=180.0, follow_redirects=True) as client:
            response = client.post(
                HF_INFERENCE_URL,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise AdDescriptionError(
            f"HuggingFace API error {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise AdDescriptionError(f"HuggingFace request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise AdDescriptionError(f"HuggingFace returned invalid JSON: {response.text[:300]}") from exc

    description = _assistant_text_from_response(data).strip()
    if not description:
        raise AdDescriptionError("HuggingFace returned an empty description.")
    return description, HF_MODEL_ID


def _describe_frame_for_ad(
    frame_paths: Sequence[Path],
    gap_duration_sec: float,
    max_words: int | None = None,
    system_prompt_prefix: str | None = None,
) -> tuple[str, str]:
    if _vision_model() == QWEN_MODEL:
        return _describe_frame_hf(frame_paths, gap_duration_sec, max_words=max_words, system_prompt_prefix=system_prompt_prefix)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AdDescriptionError("OPENAI_API_KEY is not set.")
    if not frame_paths:
        raise AdDescriptionError("At least one frame is required for AD generation.")

    resolved_word_limit = max_words or target_words(gap_duration_sec)
    prompt = "\n\n".join(
        part
        for part in (
            _frame_prompt_prefix(len(frame_paths)),
            AD_PROMPT_TEMPLATE.format(
                gap_seconds=gap_duration_sec,
                max_words=resolved_word_limit,
            ),
        )
        if part
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame_path in frame_paths:
        mime_type = mimetypes.guess_type(frame_path.name)[0] or "image/jpeg"
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encode_file_base64(frame_path)}",
                    "detail": "low",
                },
            }
        )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": content,
        }
    ]
    if system_prompt_prefix:
        messages.insert(0, {"role": "system", "content": system_prompt_prefix})
    payload = {
        "model": _vision_model(),
        "messages": messages,
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


def _frame_prompt_prefix(frame_count: int) -> str:
    if frame_count == 3:
        return (
            "You are viewing three frames from a single narration gap. "
            "Describe what is happening as a single continuous audio description."
        )
    if frame_count > 1:
        return (
            f"You are viewing {frame_count} frames from a single narration gap. "
            "Describe what is happening as a single continuous audio description."
        )
    return ""


def _vision_model() -> str:
    model = os.getenv("VN_VISION_MODEL", OPENAI_MODEL).strip() or OPENAI_MODEL
    if model not in SUPPORTED_VISION_MODELS:
        supported = ", ".join(sorted(SUPPORTED_VISION_MODELS))
        raise AdDescriptionError(f"Unsupported VN_VISION_MODEL: {model}. Supported: {supported}")
    return model


def _write_audio_file(output_path: Path, audio_bytes: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)
    if output_path.stat().st_size <= 0:
        raise AdTTSError(f"ElevenLabs returned empty audio for {output_path.name}.")


def _retry_word_limit(base_word_limit: int, retry: int) -> int:
    if retry <= 0:
        return base_word_limit
    reduced_limit = int(base_word_limit * (SHORTER_RETRY_FACTOR ** retry))
    return max(3, min(base_word_limit, reduced_limit))


def _max_allowed_audio_sec(gap_duration_sec: float) -> float:
    return max(0.0, gap_duration_sec - OVERRUN_BUFFER)


def _fit_ratio(audio_sec: float, gap_duration_sec: float) -> float:
    if gap_duration_sec <= 0:
        return 0.0
    return audio_sec / gap_duration_sec


def _truncate_to_fit(audio_bytes: bytes, gap_duration_sec: float) -> tuple[bytes, float]:
    target_sec = _max_allowed_audio_sec(gap_duration_sec)
    for attempt in range(MAX_TRUNCATION_ATTEMPTS):
        trim_target = max(0.05, target_sec - (attempt * TRUNCATION_STEP_SEC))
        truncated = truncate_audio(audio_bytes, trim_target)
        duration = audio_duration_sec(truncated)
        if fits_in_gap(duration, gap_duration_sec):
            return truncated, duration
    raise AdTTSError(
        "Unable to fit narration audio inside the detected gap after retries and truncation."
    )


def _log_audio_gap_fit(metrics: AudioGapFitMetrics) -> None:
    LOGGER.info("audio-gap-fit %s", json.dumps(metrics.json_dict(), sort_keys=True))


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
        return _vision_model()
    unique_versions = list(dict.fromkeys(model_versions))
    if len(unique_versions) == 1:
        return unique_versions[0]
    return ", ".join(unique_versions)


def _gap_midpoint(gap: GapResult) -> float:
    return gap.start_sec + (gap.duration_sec / 2)


def _gap_timestamps(gap: GapResult, frame_count: int) -> list[float]:
    mid = _gap_midpoint(gap)
    if frame_count == 1:
        return [mid]
    start = min(gap.start_sec + 0.5, mid)
    end = max(gap.end_sec - 0.5, mid)
    return [start, mid, end]


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
