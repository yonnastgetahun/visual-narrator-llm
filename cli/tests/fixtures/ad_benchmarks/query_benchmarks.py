import csv
from collections import defaultdict

CSV_PATH = "/Users/yonnasgetahun/visual-narrator-llm/cli/tests/fixtures/ad_benchmarks/cmd_ad_annotations.csv"

# NOTE: The split column uses "eval" (not "val") in this dataset.
clips = defaultdict(list)

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["split"] != "eval":
            continue
        clips[row["cmd_filename"]].append(row)

results = []

for cmd_filename, rows in clips.items():
    movie_title = rows[0]["movie_title"]
    scaled_starts = [float(r["scaled_start"]) for r in rows]
    scaled_ends = [float(r["scaled_end"]) for r in rows]
    durations = [float(r["duration"]) for r in rows]
    word_counts = [len(r["text"].split()) for r in rows]

    total_annotations = len(rows)
    time_span = max(scaled_ends) - min(scaled_starts)
    total_ad_duration = sum(durations)
    avg_word_count = sum(word_counts) / len(word_counts)
    min_scaled_end = min(scaled_ends)
    max_scaled_end = max(scaled_ends)

    results.append({
        "cmd_filename": cmd_filename,
        "movie_title": movie_title,
        "total_annotations": total_annotations,
        "time_span": time_span,
        "total_ad_duration": total_ad_duration,
        "avg_word_count": avg_word_count,
        "min_scaled_end": min_scaled_end,
        "max_scaled_end": max_scaled_end,
    })

# The eval split's clips top out at ~280s. Adjust filter range to match actual data.
# Tiers: SHORT=120-200s, MEDIUM=200-240s, LONG=240-300s
# Base quality filters kept: >=15 annotations, avg_word_count>=6
filtered = [
    r for r in results
    if 120 <= r["time_span"] <= 300
    and r["total_annotations"] >= 15
    and r["avg_word_count"] >= 6
]

# Sort by annotation count desc
filtered.sort(key=lambda x: x["total_annotations"], reverse=True)

print(f"NOTE: 'val' split not found in dataset — using 'eval' split ({len(results)} unique clips).")
print(f"Top 10 qualifying clips (span 120-300s, >=15 annotations, avg_wc>=6):\n")
print(f"{'#':<4} {'cmd_filename':<30} {'movie_title':<40} {'annots':>6} {'span(s)':>8} {'ad_dur(s)':>10} {'avg_wc':>7} {'clip_end':>9}")
print("-" * 125)
for i, r in enumerate(filtered[:10], 1):
    print(
        f"{i:<4} {r['cmd_filename']:<30} {r['movie_title'][:38]:<40} "
        f"{r['total_annotations']:>6} {r['time_span']:>8.1f} {r['total_ad_duration']:>10.1f} "
        f"{r['avg_word_count']:>7.2f} {r['max_scaled_end']:>9.1f}"
    )

print()

# Recommend 3 with span diversity across the eval dataset's actual range
short_candidates  = [r for r in filtered if r["time_span"] < 170]
medium_candidates = [r for r in filtered if 170 <= r["time_span"] < 225]
long_candidates   = [r for r in filtered if r["time_span"] >= 225]

def pick_best(candidates):
    if not candidates:
        return None
    return max(candidates, key=lambda x: x["total_annotations"])

short_pick  = pick_best(short_candidates)
medium_pick = pick_best(medium_candidates)
long_pick   = pick_best(long_candidates)

print("=== RECOMMENDED PICKS FOR VN-PROD-005 ===")
tiers = [
    ("SHORT  (<170s)",    short_pick),
    ("MEDIUM (170-225s)", medium_pick),
    ("LONG   (>=225s)",   long_pick),
]
for label, pick in tiers:
    if pick:
        print(f"\n{label}:")
        print(f"  cmd_filename:      {pick['cmd_filename']}")
        print(f"  movie_title:       {pick['movie_title']}")
        print(f"  time_span:         {pick['time_span']:.1f}s")
        print(f"  total_annotations: {pick['total_annotations']}")
        print(f"  avg_word_count:    {pick['avg_word_count']:.2f}")
        print(f"  total_ad_duration: {pick['total_ad_duration']:.1f}s")
    else:
        print(f"\n{label}: no candidates found")
