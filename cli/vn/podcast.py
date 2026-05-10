from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .api import encode_file_base64
from .compliance import analyze_compliance
from .frame import extract_frames_at
from .gaps import detect_gaps
from .output import GapResult


PODCAST_PROMPT_TEMPLATE = (
    "You are narrating a film as an audio podcast for listeners who cannot see the screen.\n"
    "Describe what is happening so the listener can follow the story as if hearing a radio drama.\n"
    "Include: what characters are doing, where they are, any significant visual story beats.\n"
    "Write in a narrative voice - present tense, vivid, immersive. Not clinical or dry.\n"
    "Maximum {max_words} words. One to three sentences. No filler phrases."
)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
GPT_COST_PER_FRAME = 0.0013
TTS_COST_PER_CHAR = 0.0003
SEGMENT_EPSILON = 0.01


class PodcastDescriptionError(RuntimeError):
    """Raised when GPT-4o frame description fails."""


class PodcastTTSError(RuntimeError):
    """Raised when ElevenLabs speech synthesis fails."""


class PodcastMixError(RuntimeError):
    """Raised when FFmpeg podcast assembly fails."""


@dataclass(frozen=True)
class PodcastNarrationEntry:
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
class PodcastResult:
    source: str
    duration_seconds: float
    gaps_found: int
    narrations_mixed: int
    output_file: str
    output_dir: str
    model_version: str
    voice_id: str
    gpt_cost_estimate: float
    tts_cost_estimate: float
    total_cost_estimate: float
    narrations: list[PodcastNarrationEntry]

    def json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "duration_seconds": round(self.duration_seconds, 3),
            "gaps_found": self.gaps_found,
            "narrations_mixed": self.narrations_mixed,
            "output_file": self.output_file,
            "output_dir": self.output_dir,
            "model_version": self.model_version,
            "voice_id": self.voice_id,
            "gpt_cost_estimate": round(self.gpt_cost_estimate, 6),
            "tts_cost_estimate": round(self.tts_cost_estimate, 6),
            "total_cost_estimate": round(self.total_cost_estimate, 6),
            "narrations": [narration.json_dict() for narration in self.narrations],
        }


def assemble_podcast(
    source: Path,
    min_gap: float = 2.0,
    voice_id: str | None = None,
    source_label: str | None = None,
    output_path: Path | None = None,
    output_dir: Path | None = None,
) -> PodcastResult:
    requested_output_dir = output_dir or Path("./vn-podcast-work")
    resolved_output_dir = requested_output_dir.expanduser()
    requested_output_path = output_path or Path("./podcast-output.mp3")
    resolved_output_path = requested_output_path.expanduser()

    audio_dir = resolved_output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    original_audio_path = resolved_output_dir / "original.mp3"
    _extract_original_audio(source, original_audio_path)

    gaps = detect_gaps(source, min_gap=min_gap)
    compliance = analyze_compliance(source, min_gap=min_gap, gaps=gaps)
    timestamps = [_gap_midpoint(gap) for gap in gaps]
    narrations: list[PodcastNarrationEntry] = []
    model_versions: list[str] = []
    resolved_voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ADAM") or DEFAULT_VOICE_ID

    if timestamps:
        with tempfile.TemporaryDirectory(prefix="vn-podcast-frames-") as tmp:
            frame_dir = Path(tmp)
            frames = extract_frames_at(source, timestamps, frame_dir)
            for index, (gap, frame) in enumerate(zip(gaps, frames, strict=True), start=1):
                description, model_version = _describe_frame_for_podcast(frame.path, gap.duration_sec)
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
                    PodcastNarrationEntry(
                        srt_index=index,
                        start_sec=gap.start_sec,
                        end_sec=gap.end_sec,
                        gap_duration_sec=gap.duration_sec,
                        gap_type=gap.gap_type,
                        frame_timestamp_sec=frame.timestamp,
                        description=description,
                        audio_file=audio_relative_path.as_posix(),
                        gpt_cost=GPT_COST_PER_FRAME,
                        tts_cost=character_count * TTS_COST_PER_CHAR,
                    )
                )

    _mix_podcast_audio(
        original_audio_path,
        gaps,
        [resolved_output_dir / narration.audio_file for narration in narrations],
        resolved_output_path,
        resolved_output_dir / "mix",
    )

    gpt_cost_estimate = sum(narration.gpt_cost for narration in narrations)
    tts_cost_estimate = sum(narration.tts_cost for narration in narrations)
    result = PodcastResult(
        source=source_label or str(source),
        duration_seconds=round(compliance.total_duration_sec, 3),
        gaps_found=len(gaps),
        narrations_mixed=len(narrations),
        output_file=_display_path(requested_output_path),
        output_dir=_display_path(requested_output_dir, directory=True),
        model_version=_combine_model_versions(model_versions),
        voice_id=resolved_voice_id,
        gpt_cost_estimate=gpt_cost_estimate,
        tts_cost_estimate=tts_cost_estimate,
        total_cost_estimate=gpt_cost_estimate + tts_cost_estimate,
        narrations=narrations,
    )

    metadata_path = resolved_output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(result.json_dict(), indent=2), encoding="utf-8")
    return result


def _mix_podcast_audio(
    original_mp3: Path,
    gaps: list[GapResult],
    narration_mp3s: list[Path],
    output_path: Path,
    tmp_dir: Path,
) -> None:
    _require_ffmpeg()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if len(gaps) != len(narration_mp3s):
        raise PodcastMixError(
            f"expected {len(gaps)} narration files for mixing, received {len(narration_mp3s)}"
        )

    if not original_mp3.exists():
        raise PodcastMixError(f"missing extracted audio: {original_mp3}")

    concat_entries: list[Path] = []
    cursor = 0.0
    original_duration = _probe_audio_duration(original_mp3)

    for index, (gap, narration_mp3) in enumerate(zip(gaps, narration_mp3s, strict=True), start=1):
        segment_start = min(max(0.0, cursor), original_duration)
        segment_end = min(max(segment_start, gap.start_sec), original_duration)
        if segment_end - segment_start > SEGMENT_EPSILON:
            segment_path = tmp_dir / f"segment_{index:05d}_before.mp3"
            _extract_segment(original_mp3, segment_start, segment_end, segment_path)
            concat_entries.append(segment_path)

        prepared_narration = tmp_dir / f"narration_{index:05d}.mp3"
        _prepare_narration_clip(narration_mp3, gap.duration_sec, prepared_narration)
        concat_entries.append(prepared_narration)
        cursor = min(max(cursor, gap.end_sec), original_duration)

    if original_duration - cursor > SEGMENT_EPSILON:
        tail_path = tmp_dir / "segment_tail.mp3"
        _extract_segment(original_mp3, cursor, original_duration, tail_path)
        concat_entries.append(tail_path)

    if not concat_entries:
        concat_entries.append(original_mp3)

    concat_file = tmp_dir / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{_escape_concat_path(path)}'\n" for path in concat_entries),
        encoding="utf-8",
    )

    _run_ffmpeg(
        [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-acodec",
            "libmp3lame",
            "-ar",
            "44100",
            "-ab",
            "192k",
            str(output_path),
            "-y",
        ],
        f"failed to assemble final podcast audio at {output_path}",
    )

    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise PodcastMixError(f"ffmpeg did not produce a usable podcast file at {output_path}")


def _extract_original_audio(source: Path, output_path: Path) -> None:
    _require_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ar",
            "44100",
            "-ab",
            "192k",
            str(output_path),
            "-y",
        ],
        f"failed to extract original audio from {source}",
    )
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise PodcastMixError(f"ffmpeg did not produce original audio at {output_path}")


def _extract_segment(original_mp3: Path, start_sec: float, end_sec: float, output_path: Path) -> None:
    if end_sec <= start_sec:
        return
    _run_ffmpeg(
        [
            "ffmpeg",
            "-i",
            str(original_mp3),
            "-ss",
            _format_seconds(start_sec),
            "-to",
            _format_seconds(end_sec),
            "-c",
            "copy",
            str(output_path),
            "-y",
        ],
        f"failed to extract original segment {start_sec:.3f}s-{end_sec:.3f}s",
    )
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise PodcastMixError(f"ffmpeg produced an empty audio segment at {output_path}")


def _prepare_narration_clip(narration_mp3: Path, gap_duration_sec: float, output_path: Path) -> None:
    if not narration_mp3.exists():
        raise PodcastMixError(f"missing narration audio: {narration_mp3}")

    narration_duration = _probe_audio_duration(narration_mp3)
    if narration_duration <= 0:
        raise PodcastMixError(f"could not determine narration duration for {narration_mp3}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if narration_duration < gap_duration_sec:
        pad_seconds = max(0.0, gap_duration_sec - narration_duration)
        _run_ffmpeg(
            [
                "ffmpeg",
                "-i",
                str(narration_mp3),
                "-af",
                f"apad=pad_dur={pad_seconds:.3f}",
                "-t",
                _format_seconds(gap_duration_sec),
                str(output_path),
                "-y",
            ],
            f"failed to pad narration clip {narration_mp3.name}",
        )
    else:
        fade_start = max(0.0, gap_duration_sec - 0.3)
        _run_ffmpeg(
            [
                "ffmpeg",
                "-i",
                str(narration_mp3),
                "-t",
                _format_seconds(gap_duration_sec),
                "-af",
                f"afade=t=out:st={fade_start:.3f}:d=0.3",
                str(output_path),
                "-y",
            ],
            f"failed to trim narration clip {narration_mp3.name}",
        )

    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise PodcastMixError(f"ffmpeg produced an empty narration segment at {output_path}")


def _describe_frame_for_podcast(frame_path: Path, gap_duration_sec: float) -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise PodcastDescriptionError("OPENAI_API_KEY is not set.")

    max_words = max(10, int(gap_duration_sec * 2.5))
    prompt = PODCAST_PROMPT_TEMPLATE.format(max_words=max_words)
    mime_type = mimetypes.guess_type(frame_path.name)[0] or "image/jpeg"
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
        "max_tokens": 200,
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
        raise PodcastDescriptionError(
            f"OpenAI API error {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise PodcastDescriptionError(f"OpenAI request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise PodcastDescriptionError(f"OpenAI returned invalid JSON: {response.text[:300]}") from exc
    if not isinstance(data, dict):
        raise PodcastDescriptionError("OpenAI returned a non-object response.")

    description = _assistant_text_from_response(data).strip()
    if not description:
        raise PodcastDescriptionError("OpenAI returned an empty description.")
    return description, _model_version_from_response(data)


def _synthesize_speech(text: str, voice_id: str, output_path: Path) -> int:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise PodcastTTSError("ELEVENLABS_API_KEY is not set.")

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
        raise PodcastTTSError(f"ElevenLabs request failed: {exc}") from exc

    if response.status_code != 200:
        raise PodcastTTSError(f"ElevenLabs API error {response.status_code}: {response.text}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    if output_path.stat().st_size <= 0:
        raise PodcastTTSError(f"ElevenLabs returned empty audio for {output_path.name}.")
    return len(text)


def _assistant_text_from_response(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise PodcastDescriptionError("OpenAI response did not include choices.")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise PodcastDescriptionError("OpenAI response did not include a valid message.")

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
    raise PodcastDescriptionError("OpenAI response content was not text.")


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


def _run_ffmpeg(command: list[str], action: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        binary = Path(command[0]).name
        raise PodcastMixError(
            f"{binary} is required for podcast mixing. Install ffmpeg so `{binary}` is available on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise PodcastMixError(f"{action}: {stderr}") from exc


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise PodcastMixError("ffmpeg is required for podcast mixing. Install it with `brew install ffmpeg`.")
    if shutil.which("ffprobe") is None:
        raise PodcastMixError("ffprobe is required for podcast mixing. Install it with `brew install ffmpeg`.")


def _probe_audio_duration(path: Path) -> float:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PodcastMixError(
            "ffprobe is required for podcast mixing. Install it with `brew install ffmpeg`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise PodcastMixError(f"ffprobe failed for {path}: {stderr}") from exc

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise PodcastMixError(f"ffprobe returned invalid JSON for {path}") from exc

    duration = ((payload.get("format") or {}).get("duration"))
    try:
        return float(duration)
    except (TypeError, ValueError):
        raise PodcastMixError(f"ffprobe did not report duration for {path}")


def _escape_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def _display_path(path: Path, directory: bool = False) -> str:
    raw = str(path)
    if path.is_absolute():
        return raw
    if raw in {".", ""}:
        return "./" if directory else "."
    if raw.startswith("./") or raw.startswith("../"):
        return raw
    return f"./{raw}"


def _format_seconds(value: float) -> str:
    return f"{max(0.0, value):.3f}"
