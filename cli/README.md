# Visual Narrator CLI

`visual-narrator` installs the `vn` command for generating audio description text from local video files, image frames, or YouTube URLs using the live Visual Narrator Frame Description API.

## Install

```bash
pip install visual-narrator
```

For local development from this repository:

```bash
cd cli
pip install -e .
```

The CLI uses `ffmpeg` for video frame extraction. Install it separately if it is not already available:

```bash
brew install ffmpeg
```

## Authentication

Pass an API key with `--api-key` or set `VN_API_KEY`:

```bash
export VN_API_KEY=vn_live_your_key
```

Create a free-tier key:

```bash
vn keys create dev@example.com
```

## Describe A Video

Local video:

```bash
vn describe ./demo.mp4 --api-key "$VN_API_KEY" --format json
```

YouTube URL:

```bash
vn describe "https://youtube.com/watch?v=VIDEO_ID" --api-key "$VN_API_KEY" --format srt
```

Single image frame:

```bash
vn describe /tmp/vn-test.jpg --api-key "$VN_API_KEY" --format text
```

Sampling defaults to one frame every three seconds. Configure it with `--fps`:

```bash
vn describe ./demo.mp4 --fps 1 --format json
```

Override the API base URL:

```bash
vn describe ./demo.mp4 --api-url http://localhost:3000 --api-key "$VN_API_KEY"
```

## Detect Narration Gaps

Find silence windows and dialogue breaks where narration can fit:

```bash
vn gaps ./demo.mp4 --format json
```

YouTube URLs use the same download path as `vn describe`:

```bash
vn gaps "https://youtube.com/watch?v=VIDEO_ID" --format text --min-gap 2.5
```

Choose the Whisper model with `--whisper-model` when you want a faster or more accurate pass:

```bash
vn gaps ./demo.mp4 --whisper-model base --format srt
```

Output formats:

```bash
vn gaps ./demo.mp4 --format json
vn gaps ./demo.mp4 --format text
vn gaps ./demo.mp4 --format srt
```

JSON output returns objects with `start_sec`, `end_sec`, `duration_sec`, and `gap_type`.

## Compliance Reports

Score a video against WCAG/CVAA audio-description checks using the same gap detector:

```bash
vn compliance ./demo.mp4 --format json
```

Text output is available for quick terminal review:

```bash
vn compliance ./demo.mp4 --format text --min-gap 3.0
```

Choose the Whisper model the same way as `vn gaps`:

```bash
vn compliance ./demo.mp4 --whisper-model base --format json
```

JSON output returns:

```json
{
  "score": 67,
  "wcag_level": "A",
  "criteria": {
    "wcag_1_2_3": {"passed": true},
    "wcag_1_2_5": {"passed": false},
    "cvaa_audio_description": {"passed": true}
  },
  "gaps": [],
  "recommendations": []
}
```

Coverage is calculated from silence and music-only gap duration divided by total video duration. Recommendations are capped at 10 narration opportunities.

## Output Formats

JSON:

```bash
vn describe ./demo.mp4 --format json
```

Returns:

```json
[
  {
    "timecode": "00:00:00.000",
    "description": "A person walks through a city street.",
    "objects_detected": [{"label": "Person", "confidence": 98.7}],
    "latency_ms": 241
  }
]
```

SRT:

```bash
vn describe ./demo.mp4 --format srt
```

Text:

```bash
vn describe ./demo.mp4 --format text
```

## Benchmark

Run a single-frame benchmark comparing Visual Narrator, GPT-4o, and Gemini:

```bash
vn benchmark /tmp/vn-test.jpg
```

Use a custom API deployment:

```bash
vn benchmark /tmp/vn-test.jpg --api-url http://localhost:3000
```

## Commands

```bash
vn --help
vn describe --help
vn gaps --help
vn compliance --help
vn benchmark --help
vn keys create --help
```
