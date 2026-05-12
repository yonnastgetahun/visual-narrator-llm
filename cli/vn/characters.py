"""Character context for audio description narration.

Two sources (combined when both available):
  1. Upfront character sheet  — filmmaker uploads a .txt file
  2. Auto-detected from audio — Whisper transcribes dialogue, names extracted live

The roster accumulates as the film progresses so names heard later in the
film are available for subsequent gaps.
"""
from __future__ import annotations

import os
import re
import httpx
from dataclasses import dataclass, field
from pathlib import Path


_WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"

# Words that look like proper nouns but aren't character names
_STOPWORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "american", "english", "french", "german", "british", "chinese",
    "mr", "mrs", "ms", "dr", "sir", "god", "lord",
}


@dataclass
class CharacterRoster:
    """Tracks known character names and formats them for the AD prompt."""

    # name -> optional description ("young male protagonist")
    names: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "CharacterRoster":
        """Load from a character sheet .txt file.

        Accepted formats (one entry per line):
          Nathan
          Nathan: young male protagonist
          Nathan (young male protagonist)
        Blank lines and lines starting with # are ignored.
        """
        roster = cls()
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # "Name: description" or "Name (description)"
            m = re.match(r"([A-Za-z][A-Za-z\s'.\-]+?)(?:[:\(]\s*(.+?)[\)]?)?$", line)
            if m:
                name = m.group(1).strip()
                desc = (m.group(2) or "").strip().rstrip(")")
                roster.names[name] = desc
        return roster

    @classmethod
    def from_transcript(cls, transcript: str) -> "CharacterRoster":
        """Extract likely character names from a Whisper transcript.

        Heuristics:
        - Direct address: "Nathan, stop!" / "Come on, Karen."
        - Repeated capitalised words (3+ occurrences) that aren't stopwords
        """
        roster = cls()

        # Vocative pattern: word followed by comma at sentence boundary,
        # or comma followed by word at sentence boundary
        vocative = re.findall(
            r"(?:^|[.!?]\s+)([A-Z][a-z]{2,})(?=,)|,\s*([A-Z][a-z]{2,})(?=[.!?]|\s)",
            transcript,
        )
        for groups in vocative:
            for name in groups:
                if name and name.lower() not in _STOPWORDS:
                    roster.names.setdefault(name, "")

        # Frequency pass: capitalised words appearing 3+ times
        candidates = re.findall(r"\b([A-Z][a-z]{2,})\b", transcript)
        freq: dict[str, int] = {}
        for w in candidates:
            if w.lower() not in _STOPWORDS:
                freq[w] = freq.get(w, 0) + 1
        for name, count in freq.items():
            if count >= 3:
                roster.names.setdefault(name, "")

        return roster

    def merge(self, other: "CharacterRoster") -> "CharacterRoster":
        """Return a new roster combining both (self takes priority on descriptions)."""
        merged = CharacterRoster(names=dict(other.names))
        for name, desc in self.names.items():
            merged.names[name] = desc or merged.names.get(name, "")
        return merged

    def update_from_transcript(self, transcript: str) -> None:
        """Add any newly discovered names from a dialogue segment in-place."""
        discovered = CharacterRoster.from_transcript(transcript)
        for name, desc in discovered.names.items():
            self.names.setdefault(name, desc)

    def to_prompt_fragment(self) -> str | None:
        """Format names for injection into the AD system prompt."""
        if not self.names:
            return None
        parts = []
        for name, desc in self.names.items():
            parts.append(f"{name}{f' ({desc})' if desc else ''}")
        return "Characters in this film: " + ", ".join(parts) + ". Use these names."

    def is_empty(self) -> bool:
        return not self.names


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe an audio/video file with OpenAI Whisper and return the text."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return ""
    with httpx.Client(timeout=120) as client:
        with open(audio_path, "rb") as f:
            resp = client.post(
                _WHISPER_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                data={"model": "whisper-1"},
                files={"file": (audio_path.name, f, "audio/mpeg")},
            )
        resp.raise_for_status()
        return resp.json().get("text", "")


def build_roster_from_audio(audio_path: Path) -> CharacterRoster:
    """Transcribe audio and return a CharacterRoster of detected names."""
    transcript = transcribe_audio(audio_path)
    if not transcript:
        return CharacterRoster()
    return CharacterRoster.from_transcript(transcript)
