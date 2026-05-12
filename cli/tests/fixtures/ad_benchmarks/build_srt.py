from __future__ import annotations

import csv
from pathlib import Path


CLIP_IDS = ("6nvQySlxSWY", "7ELmyf41TnQ", "v0KOJWyR4SU")
FIXTURE_ROOT = Path(__file__).resolve().parent
CSV_PATH = FIXTURE_ROOT / "cmd_ad_annotations.csv"
CLIPS_DIR = FIXTURE_ROOT / "clips"


def _format_srt_time(seconds: float) -> str:
    total_millis = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _rows_for_clip(rows: list[dict[str, str]], clip_id: str) -> list[dict[str, str]]:
    return sorted(
        (
            row
            for row in rows
            if row["cmd_filename"].endswith(clip_id) and row["split"] == "eval"
        ),
        key=lambda row: float(row["scaled_start"]),
    )


def _write_srt(clip_id: str, rows: list[dict[str, str]]) -> Path:
    output_path = CLIPS_DIR / f"{clip_id}_professional.srt"
    blocks: list[str] = []
    for index, row in enumerate(rows, start=1):
        start = _format_srt_time(float(row["scaled_start"]))
        end = _format_srt_time(float(row["scaled_end"]))
        text = row["text"].strip()
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    output_path.write_text("\n\n".join(blocks), encoding="utf-8")
    return output_path


def main() -> None:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for clip_id in CLIP_IDS:
        matched_rows = _rows_for_clip(rows, clip_id)
        if not matched_rows:
            raise SystemExit(f"No eval rows found for clip {clip_id}")
        output_path = _write_srt(clip_id, matched_rows)
        print(output_path)


if __name__ == "__main__":
    main()
