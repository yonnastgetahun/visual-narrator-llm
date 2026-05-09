from __future__ import annotations

import os
import json
import importlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .output import GapResult


SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<seconds>\d+(?:\.\d+)?)")
SILENCE_END_RE = re.compile(r"silence_end:\s*(?P<seconds>\d+(?:\.\d+)?)")


class GapDetectionError(RuntimeError):
    """Raised when narration gaps cannot be detected."""


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def detect_gaps(source: Path, whisper_model: str = "base", min_gap: float = 2.0) -> list[GapResult]:
    source = source.expanduser().resolve()
    if min_gap <= 0:
        raise GapDetectionError("--min-gap must be greater than 0")
    if not source.exists():
        raise GapDetectionError(f"input does not exist: {source}")
    if source.is_dir():
        raise GapDetectionError(f"input must be a file, not a directory: {source}")
    if shutil.which("ffmpeg") is None:
        raise GapDetectionError("ffmpeg is required for gap detection. Install it with `brew install ffmpeg`.")

    duration, has_audio = _probe_media(source)
    if duration <= 0:
        raise GapDetectionError(f"could not determine media duration for {source}")

    if not has_audio:
        return _full_media_gap(duration, min_gap)

    with tempfile.TemporaryDirectory(prefix="vn-gaps-") as tmp:
        tmp_path = Path(tmp)
        audio_path = tmp_path / "audio.wav"
        _extract_audio(source, audio_path)
        silences = _detect_silences(source, duration, min_gap)
        transcription = _transcribe_audio(audio_path, whisper_model, _whisper_model_dir())

    words = _collect_words(transcription)
    segments = _collect_segments(transcription)
    candidates = _build_candidates(words, duration)
    if not candidates and duration >= min_gap:
        candidates = [Interval(0.0, duration)]

    gaps: list[GapResult] = []
    for candidate in candidates:
        if candidate.duration < min_gap:
            continue
        gap_type = _classify_gap(candidate, silences, segments)
        gaps.append(
            GapResult(
                start_sec=candidate.start,
                end_sec=candidate.end,
                duration_sec=candidate.duration,
                gap_type=gap_type,
            )
        )
    return gaps


def _full_media_gap(duration: float, min_gap: float) -> list[GapResult]:
    if duration < min_gap:
        return []
    return [GapResult(start_sec=0.0, end_sec=duration, duration_sec=duration, gap_type="silence")]


def _probe_media(source: Path) -> tuple[float, bool]:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        probe = json.loads(completed.stdout or "{}")
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise GapDetectionError(f"ffmpeg probe failed for {source}: {stderr}") from exc

    duration = _duration_from_probe(probe)
    has_audio = any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))
    return duration, has_audio


def _duration_from_probe(probe: dict[str, Any]) -> float:
    format_info = probe.get("format") or {}
    duration = format_info.get("duration")
    if duration is not None:
        try:
            return float(duration)
        except (TypeError, ValueError):
            pass

    stream_durations: list[float] = []
    for stream in probe.get("streams", []):
        value = stream.get("duration")
        if value is None:
            continue
        try:
            stream_durations.append(float(value))
        except (TypeError, ValueError):
            continue
    if stream_durations:
        return max(stream_durations)
    return 0.0


def _extract_audio(source: Path, output_path: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise GapDetectionError(f"ffmpeg failed to extract audio: {stderr}") from exc

    if not output_path.exists():
        raise GapDetectionError(f"ffmpeg did not produce an audio file for {source}")


def _detect_silences(source: Path, duration: float, min_gap: float) -> list[Interval]:
    silence_floor = "30dB"
    silence_duration = max(0.25, min(0.75, min_gap / 2))
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-af",
        f"silencedetect=noise=-{silence_floor}:d={silence_duration}",
        "-f",
        "null",
        "-",
    ]

    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise GapDetectionError(f"ffmpeg silencedetect failed: {stderr}") from exc

    stderr = completed.stderr or ""
    intervals: list[Interval] = []
    pending_start: float | None = None
    for line in stderr.splitlines():
        start_match = SILENCE_START_RE.search(line)
        if start_match:
            pending_start = float(start_match.group("seconds"))
            continue
        end_match = SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            intervals.append(Interval(start=pending_start, end=float(end_match.group("seconds"))))
            pending_start = None

    if pending_start is not None:
        intervals.append(Interval(start=pending_start, end=duration))
    return _merge_intervals(intervals)


def _whisper_model_dir() -> Path:
    model_dir = Path(
        os.getenv("VN_WHISPER_MODEL_DIR") or Path(tempfile.gettempdir()) / "vn-whisper-models"
    ).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _transcribe_audio(audio_path: Path, whisper_model: str, model_dir: Path) -> dict[str, Any]:
    try:
        whisper = importlib.import_module("whisper")
    except ImportError as exc:
        return _transcribe_with_cli(audio_path, whisper_model, model_dir)

    try:
        model = whisper.load_model(whisper_model, download_root=str(model_dir))
    except Exception as exc:  # noqa: BLE001
        raise GapDetectionError(f"failed to load Whisper model '{whisper_model}': {exc}") from exc

    try:
        import io
        import sys as _sys
        _old_stdout = _sys.stdout
        _sys.stdout = io.StringIO()
        try:
            result = model.transcribe(str(audio_path), word_timestamps=True, verbose=False)
        finally:
            _sys.stdout = _old_stdout
        return result
    except Exception as exc:  # noqa: BLE001
        raise GapDetectionError(f"Whisper transcription failed: {exc}") from exc


def _transcribe_with_cli(audio_path: Path, whisper_model: str, model_dir: Path) -> dict[str, Any]:
    whisper_bin = shutil.which("whisper")
    if whisper_bin is None:
        raise GapDetectionError(
            "Whisper is required for gap detection. Install openai-whisper or make the `whisper` CLI available."
        )

    with tempfile.TemporaryDirectory(prefix="vn-whisper-") as output_dir:
        try:
            subprocess.run(
                [
                    whisper_bin,
                    str(audio_path),
                    "--model",
                    whisper_model,
                    "--output_format",
                    "json",
                    "--output_dir",
                    output_dir,
                    "--model_dir",
                    str(model_dir),
                    "--word_timestamps",
                    "True",
                    "--fp16",
                    "False",
                    "--verbose",
                    "False",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
            raise GapDetectionError(f"Whisper CLI transcription failed: {stderr}") from exc

        json_path = Path(output_dir) / f"{audio_path.stem}.json"
        if not json_path.exists():
            raise GapDetectionError("Whisper CLI completed but did not produce a JSON transcript")

        try:
            return json.loads(json_path.read_text())
        except json.JSONDecodeError as exc:
            raise GapDetectionError(f"Whisper CLI produced invalid JSON: {exc}") from exc


# Segments with no_speech_prob above this threshold are likely hallucinated
# (gunshots, music, etc.) and are excluded from speech word collection.
_NO_SPEECH_PROB_THRESHOLD = 0.35

# Words with probability below this threshold inside a valid speech segment
# are treated as hallucinated and excluded from candidate building.
_WORD_PROB_THRESHOLD = 0.30


def _collect_words(transcription: dict[str, Any]) -> list[Interval]:
    words: list[Interval] = []
    for segment in transcription.get("segments", []):
        # Skip segments Whisper itself flagged as likely non-speech
        no_speech_prob = segment.get("no_speech_prob")
        if no_speech_prob is not None:
            try:
                if float(no_speech_prob) > _NO_SPEECH_PROB_THRESHOLD:
                    continue
            except (TypeError, ValueError):
                pass
        for word in segment.get("words", []) or []:
            start = word.get("start")
            end = word.get("end")
            prob = word.get("probability")
            if start is None or end is None:
                continue
            # Skip low-confidence words (hallucinations from non-speech audio)
            if prob is not None:
                try:
                    if float(prob) < _WORD_PROB_THRESHOLD:
                        continue
                except (TypeError, ValueError):
                    pass
            try:
                words.append(Interval(start=float(start), end=float(end)))
            except (TypeError, ValueError):
                continue
    return sorted(words, key=lambda item: (item.start, item.end))


def _collect_segments(transcription: dict[str, Any]) -> list[Interval]:
    segments: list[Interval] = []
    for segment in transcription.get("segments", []):
        # Exclude segments Whisper flagged as likely non-speech
        no_speech_prob = segment.get("no_speech_prob")
        if no_speech_prob is not None:
            try:
                if float(no_speech_prob) > _NO_SPEECH_PROB_THRESHOLD:
                    continue
            except (TypeError, ValueError):
                pass
        start = segment.get("start")
        end = segment.get("end")
        if start is None or end is None:
            continue
        try:
            segments.append(Interval(start=float(start), end=float(end)))
        except (TypeError, ValueError):
            continue
    return sorted(segments, key=lambda item: (item.start, item.end))


def _build_candidates(words: list[Interval], duration: float) -> list[Interval]:
    if not words:
        return []

    candidates: list[Interval] = []
    first_word = words[0]
    if first_word.start > 0:
        candidates.append(Interval(start=0.0, end=first_word.start))

    previous = words[0]
    for current in words[1:]:
        if current.start > previous.end:
            candidates.append(Interval(start=previous.end, end=current.start))
        previous = current

    if previous.end < duration:
        candidates.append(Interval(start=previous.end, end=duration))
    return _merge_intervals(candidates)


def _classify_gap(candidate: Interval, silences: list[Interval], segments: list[Interval]) -> str:
    if candidate.duration <= 0:
        return "silence"

    silence_overlap = _coverage(candidate, silences)
    if silence_overlap / candidate.duration >= 0.8:
        return "silence"

    for segment in segments:
        if candidate.start >= segment.start and candidate.end <= segment.end:
            return "speech"

    return "music_only"


def _coverage(candidate: Interval, intervals: Iterable[Interval]) -> float:
    total = 0.0
    for interval in intervals:
        overlap_start = max(candidate.start, interval.start)
        overlap_end = min(candidate.end, interval.end)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []

    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    merged: list[Interval] = [ordered[0]]
    for interval in ordered[1:]:
        last = merged[-1]
        if interval.start <= last.end:
            merged[-1] = Interval(start=last.start, end=max(last.end, interval.end))
        else:
            merged.append(interval)
    return merged


def _decode_ffmpeg_error(exc: Exception) -> str:
    stderr = getattr(exc, "stderr", b"")
    stdout = getattr(exc, "stdout", b"")
    payload = stderr or stdout or b""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace").strip()
    return str(payload).strip()
