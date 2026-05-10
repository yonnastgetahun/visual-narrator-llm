from __future__ import annotations

import json
import mimetypes
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import httpx
from botocore.exceptions import BotoCoreError, ClientError

from .api import encode_file_base64
from .frame import extract_frames
from .gaps import GapDetectionError, _probe_media


SPORTS_PROMPT_TEMPLATE = (
    "You are a live audio describer for a blind sports audience. Describe the action you see in plain, "
    "active present tense - the way a commentator would call a play on radio.\n\n"
    "Tracked objects in this frame (positions are normalized 0-1, origin top-left):\n"
    "{tracker_state_json}\n\n"
    "Be specific about what tracked objects are doing and where they are on the field/court. "
    "If you can infer the sport, use sport-specific language (e.g. \"drives to the basket\", "
    "\"crosses into the box\"). One to two sentences. No filler. Call the action."
)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o"
GPT_COST_PER_FRAME = 0.0013
REKOGNITION_COST_PER_IMAGE = 0.001
SPORTS_RELEVANT_LABELS = {
    "Person",
    "Ball",
    "Sports",
    "Soccer",
    "Football",
    "Basketball",
    "Baseball",
    "Referee",
    "Crowd",
    "Goal Post",
    "Court",
    "Field",
    "Athlete",
    "Player",
    "Stadium",
}


class SportsDetectionError(RuntimeError):
    """Raised when sports detection or narration fails."""


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    label: str
    confidence: float
    center_x: float
    center_y: float
    width: float
    height: float

    def json_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "center_x": round(self.center_x, 4),
            "center_y": round(self.center_y, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
        }


@dataclass(frozen=True)
class SportsNarrationEntry:
    srt_index: int
    timestamp_sec: float
    tracked_objects: list[TrackedObject]
    narration: str
    gpt_cost: float

    def json_dict(self) -> dict[str, Any]:
        return {
            "srt_index": self.srt_index,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "tracked_objects": [obj.json_dict() for obj in self.tracked_objects],
            "narration": self.narration,
            "gpt_cost": round(self.gpt_cost, 6),
        }


@dataclass(frozen=True)
class SportsResult:
    source: str
    duration_seconds: float
    frames_analyzed: int
    narrations_generated: int
    fps: float
    narrate_every_sec: float
    model_version: str
    rekognition_cost_estimate: float
    gpt_cost_estimate: float
    total_cost_estimate: float
    narrations: list[SportsNarrationEntry]

    def json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "duration_seconds": round(self.duration_seconds, 3),
            "frames_analyzed": self.frames_analyzed,
            "narrations_generated": self.narrations_generated,
            "fps": round(self.fps, 3),
            "narrate_every_sec": round(self.narrate_every_sec, 3),
            "model_version": self.model_version,
            "rekognition_cost_estimate": round(self.rekognition_cost_estimate, 6),
            "gpt_cost_estimate": round(self.gpt_cost_estimate, 6),
            "total_cost_estimate": round(self.total_cost_estimate, 6),
            "narrations": [narration.json_dict() for narration in self.narrations],
        }


@dataclass
class _TrackState:
    track_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    missed: int = 0

    def update(self, detection: Detection) -> None:
        self.label = detection.label
        self.confidence = detection.confidence
        self.x1 = detection.x1
        self.y1 = detection.y1
        self.x2 = detection.x2
        self.y2 = detection.y2
        self.missed = 0

    def to_tracked_object(self) -> TrackedObject:
        width = max(0.0, self.x2 - self.x1)
        height = max(0.0, self.y2 - self.y1)
        return TrackedObject(
            track_id=self.track_id,
            label=self.label,
            confidence=self.confidence,
            center_x=self.x1 + (width / 2),
            center_y=self.y1 + (height / 2),
            width=width,
            height=height,
        )


class IoUTracker:
    """
    Assigns stable track IDs to detected bounding boxes across frames using IoU matching.
    Tracks not matched for max_age consecutive frames are dropped.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_track_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        assignments: dict[int, int] = {}
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        candidates: list[tuple[float, int, int]] = []

        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                if track.label != detection.label:
                    continue
                iou = _intersection_over_union(track, detection)
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_id, detection_index))

        for _iou, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            self._tracks[track_id].update(detections[detection_index])
            assignments[detection_index] = track_id
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue
            track.missed += 1
            if track.missed > self.max_age:
                del self._tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                label=detection.label,
                confidence=detection.confidence,
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
            )
            assignments[detection_index] = track_id

        return [
            self._tracks[assignments[index]].to_tracked_object()
            for index in range(len(detections))
            if index in assignments
        ]


def assemble_sports_kit(
    source: Path,
    fps: float = 2.0,
    narrate_every: float = 4.0,
    min_confidence: float = 0.7,
    source_label: str | None = None,
    output_dir: Path | None = None,
) -> SportsResult:
    source = source.expanduser().resolve()
    _validate_environment()

    if fps <= 0:
        raise SportsDetectionError("--fps must be greater than 0")
    if narrate_every <= 0:
        raise SportsDetectionError("--narrate-every must be greater than 0")
    if not 0 < min_confidence <= 1:
        raise SportsDetectionError("--min-confidence must be between 0 and 1")

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    rekognition_client = _create_rekognition_client()
    frame_dir = output_dir or (source.parent / ".vn-sports-frames")
    frames = extract_frames(source, frame_dir, fps=fps)
    duration_seconds = _media_duration(source)
    tracker = IoUTracker()
    narrations: list[SportsNarrationEntry] = []
    model_versions: list[str] = []
    frame_interval = 1.0 / fps

    for frame in frames:
        detections = _detect_objects(frame.path, rekognition_client, min_confidence=min_confidence)
        tracked_objects = tracker.update(detections)
        if not _should_narrate(frame.timestamp, narrate_every=narrate_every, frame_interval=frame_interval):
            continue

        narration, model_version = _narrate_frame(frame.path, tracked_objects, openai_api_key=openai_api_key)
        if model_version:
            model_versions.append(model_version)
        narrations.append(
            SportsNarrationEntry(
                srt_index=len(narrations) + 1,
                timestamp_sec=frame.timestamp,
                tracked_objects=tracked_objects,
                narration=narration,
                gpt_cost=GPT_COST_PER_FRAME,
            )
        )

    frames_analyzed = len(frames)
    narrations_generated = len(narrations)
    rekognition_cost = frames_analyzed * REKOGNITION_COST_PER_IMAGE
    gpt_cost = narrations_generated * GPT_COST_PER_FRAME

    return SportsResult(
        source=source_label or str(source),
        duration_seconds=duration_seconds,
        frames_analyzed=frames_analyzed,
        narrations_generated=narrations_generated,
        fps=fps,
        narrate_every_sec=narrate_every,
        model_version=_combine_model_versions(model_versions),
        rekognition_cost_estimate=rekognition_cost,
        gpt_cost_estimate=gpt_cost,
        total_cost_estimate=rekognition_cost + gpt_cost,
        narrations=narrations,
    )


def _validate_environment() -> None:
    required = {
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "").strip(),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", "").strip(),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "").strip(),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SportsDetectionError(f"Missing required environment variable(s): {', '.join(missing)}")


def _create_rekognition_client() -> Any:
    try:
        return boto3.client("rekognition")
    except (BotoCoreError, ClientError) as exc:
        raise SportsDetectionError(f"Failed to create Rekognition client: {exc}") from exc


def _detect_objects(frame_path: Path, rekognition_client: Any, min_confidence: float) -> list[Detection]:
    try:
        response = rekognition_client.detect_labels(
            Image={"Bytes": frame_path.read_bytes()},
            MaxLabels=20,
            MinConfidence=min_confidence * 100.0,
        )
    except ClientError as exc:
        raise SportsDetectionError(f"Rekognition detect_labels failed: {exc}") from exc
    except BotoCoreError as exc:
        raise SportsDetectionError(f"Rekognition request failed: {exc}") from exc

    metadata = response.get("ResponseMetadata") if isinstance(response, dict) else None
    status_code = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    if status_code != 200:
        raise SportsDetectionError(f"Rekognition returned HTTP status {status_code!r} for {frame_path.name}")

    detections: list[Detection] = []
    for label in response.get("Labels", []):
        if not isinstance(label, dict):
            continue
        label_name = str(label.get("Name") or "").strip()
        if label_name not in SPORTS_RELEVANT_LABELS:
            continue
        instances = label.get("Instances")
        if not isinstance(instances, list):
            continue
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            bbox = instance.get("BoundingBox")
            if not isinstance(bbox, dict):
                continue
            confidence = _normalized_confidence(instance.get("Confidence"))
            if confidence < min_confidence:
                continue
            x1 = _bounded_float(bbox.get("Left"))
            y1 = _bounded_float(bbox.get("Top"))
            width = _bounded_float(bbox.get("Width"))
            height = _bounded_float(bbox.get("Height"))
            detections.append(
                Detection(
                    label=label_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=min(1.0, x1 + width),
                    y2=min(1.0, y1 + height),
                )
            )
    return detections


def _narrate_frame(
    frame_path: Path,
    tracked_objects: Iterable[TrackedObject],
    openai_api_key: str,
) -> tuple[str, str]:
    tracker_state_json = json.dumps([obj.json_dict() for obj in tracked_objects], indent=2)
    prompt = SPORTS_PROMPT_TEMPLATE.format(tracker_state_json=tracker_state_json)
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
        "max_tokens": 150,
    }

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.post(
                OPENAI_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json",
                },
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise SportsDetectionError(
            f"OpenAI API error {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise SportsDetectionError(f"OpenAI request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise SportsDetectionError(f"OpenAI returned invalid JSON: {response.text[:300]}") from exc

    if not isinstance(data, dict):
        raise SportsDetectionError("OpenAI returned a non-object response.")

    narration = _assistant_text_from_response(data).strip()
    if not narration:
        raise SportsDetectionError("OpenAI returned an empty narration.")
    return narration, _model_version_from_response(data)


def _should_narrate(timestamp: float, narrate_every: float, frame_interval: float) -> bool:
    remainder = timestamp % narrate_every
    return remainder < frame_interval or abs(remainder - narrate_every) < 1e-9


def _media_duration(source: Path) -> float:
    try:
        duration, _has_audio = _probe_media(source)
    except GapDetectionError as exc:
        raise SportsDetectionError(str(exc)) from exc
    if duration <= 0:
        raise SportsDetectionError(f"could not determine media duration for {source}")
    return round(duration, 3)


def _combine_model_versions(model_versions: list[str]) -> str:
    if not model_versions:
        return OPENAI_MODEL
    unique_versions = list(dict.fromkeys(model_versions))
    if len(unique_versions) == 1:
        return unique_versions[0]
    return ", ".join(unique_versions)


def _assistant_text_from_response(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SportsDetectionError("OpenAI response did not include choices.")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise SportsDetectionError("OpenAI response did not include a valid message.")

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
    raise SportsDetectionError("OpenAI response content was not text.")


def _model_version_from_response(data: dict[str, Any]) -> str:
    model = data.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return ""


def _normalized_confidence(value: Any) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    if raw > 1.0:
        raw /= 100.0
    return max(0.0, min(1.0, raw))


def _bounded_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _intersection_over_union(track: _TrackState, detection: Detection) -> float:
    inter_left = max(track.x1, detection.x1)
    inter_top = max(track.y1, detection.y1)
    inter_right = min(track.x2, detection.x2)
    inter_bottom = min(track.y2, detection.y2)
    inter_width = max(0.0, inter_right - inter_left)
    inter_height = max(0.0, inter_bottom - inter_top)
    intersection = inter_width * inter_height
    if intersection <= 0:
        return 0.0

    track_area = max(0.0, track.x2 - track.x1) * max(0.0, track.y2 - track.y1)
    detection_area = max(0.0, detection.x2 - detection.x1) * max(0.0, detection.y2 - detection.y1)
    union = track_area + detection_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union
