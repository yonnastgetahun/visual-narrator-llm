#!/usr/bin/env python3
"""E2E smoke test for the Visual Narrator Demo API on Railway."""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from urllib.parse import urlencode

BASE_URL = "https://visual-narrator-llm-production.up.railway.app"

# Short public-domain Wikimedia clip (~30 s) — no yt-dlp needed, pure direct download.
TEST_VIDEO = "https://media.w3.org/2010/05/sintel/trailer.mp4"  # Sintel trailer, ~6MB, 52s, W3C test file


def _print(msg: str, *, ok: bool = True) -> None:
    tag = "OK " if ok else "ERR"
    print(f"[{tag}] {msg}", flush=True)


def check_health() -> bool:
    url = f"{BASE_URL}/health"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            body = json.loads(resp.read())
            if body.get("status") == "ok":
                _print(f"GET /health → {body}")
                return True
            _print(f"GET /health unexpected body: {body}", ok=False)
            return False
    except urllib.error.HTTPError as exc:
        _print(f"GET /health HTTP {exc.code}: {exc.reason}", ok=False)
    except Exception as exc:
        _print(f"GET /health error: {exc}", ok=False)
    return False


def stream_ad(source: str, min_gap: float = 4.0) -> dict | None:
    """Stream the AD endpoint and return the manifest on success."""
    params = urlencode({"source": source, "min_gap": min_gap})
    url = f"{BASE_URL}/api/ad?{params}"
    print(f"\n[   ] Streaming GET /api/ad  min_gap={min_gap}s", flush=True)
    print(f"[   ] source: {source}\n", flush=True)

    manifest = None
    try:
        req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            buffer = ""
            event_name = None
            for raw in resp:
                line = raw.decode("utf-8", errors="replace")
                buffer += line

                while "\n" in buffer:
                    line_part, buffer = buffer.split("\n", 1)
                    line_part = line_part.rstrip("\r")

                    if line_part.startswith("event:"):
                        event_name = line_part[len("event:"):].strip()
                    elif line_part.startswith("data:"):
                        data_str = line_part[len("data:"):].strip()
                        try:
                            payload = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        step = payload.get("step", event_name or "?")

                        if event_name == "step":
                            msg = payload.get("message", "")
                            gaps = payload.get("gaps")
                            dur = payload.get("duration_seconds")
                            extra = ""
                            if gaps is not None:
                                extra = f"  gaps={gaps}  duration={dur}s"
                            _print(f"{step:15s}  {msg}{extra}")

                        elif event_name == "complete":
                            manifest = payload.get("manifest", {})
                            gaps_found = manifest.get("gaps_found", 0)
                            cost = manifest.get("total_cost_estimate", 0)
                            duration = manifest.get("duration_seconds", 0)
                            narrations = manifest.get("narrations", [])
                            audio_count = sum(1 for n in narrations if n.get("audio_data"))
                            _print(
                                f"complete         "
                                f"gaps={gaps_found}  duration={duration}s  "
                                f"audio={audio_count}/{gaps_found}  cost=${cost:.4f}"
                            )
                            # Spot-check first narration
                            if narrations:
                                first = narrations[0]
                                desc_preview = first.get("description", "")[:80].replace("\n", " ")
                                _print(f"  first desc    → \"{desc_preview}...\"")

                        elif event_name == "error":
                            _print(f"API error: {payload.get('message')}", ok=False)
                            return None

                        event_name = None

    except urllib.error.HTTPError as exc:
        _print(f"GET /api/ad HTTP {exc.code}: {exc.reason}", ok=False)
    except Exception as exc:
        _print(f"GET /api/ad error: {exc}", ok=False)

    return manifest


def main() -> int:
    source = sys.argv[1] if len(sys.argv) > 1 else TEST_VIDEO

    print(f"\n{'='*60}")
    print("Visual Narrator Demo API — E2E Smoke Test")
    print(f"{'='*60}\n")

    if not check_health():
        print("\nHealth check failed — Railway may not be up yet. Try again in ~60s.")
        return 1

    manifest = stream_ad(source)
    if manifest is None:
        print("\nAD stream failed.")
        return 1

    print(f"\n{'='*60}")
    print("PASS — end-to-end AD generation completed successfully.")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
