import type { Manifest, NarrationEntry } from "@/lib/types";

function splitTime(seconds: number) {
  const totalMs = Math.max(0, Math.round(seconds * 1000));
  const millis = totalMs % 1000;
  const totalSeconds = Math.floor(totalMs / 1000);
  const secs = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const minutes = totalMinutes % 60;
  const hours = Math.floor(totalMinutes / 60);
  return { hours, minutes, secs, millis };
}

export function formatClock(seconds: number) {
  const { hours, minutes, secs } = splitTime(seconds);
  return [hours, minutes, secs].map((value) => String(value).padStart(2, "0")).join(":");
}

export function formatSrtTime(seconds: number) {
  const { hours, minutes, secs, millis } = splitTime(seconds);
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")},${String(millis).padStart(3, "0")}`;
}

export function formatDuration(seconds: number) {
  return `${seconds.toFixed(1)}s`;
}

export function buildSrt(narrations: NarrationEntry[]) {
  return narrations
    .map(
      (entry) =>
        `${entry.srt_index}\n${formatSrtTime(entry.start_sec)} --> ${formatSrtTime(entry.end_sec)}\n${entry.description}`,
    )
    .join("\n\n");
}

export function manifestWithoutAudio(manifest: Manifest) {
  return {
    ...manifest,
    narrations: manifest.narrations.map(({ audio_data, ...rest }) => rest),
  };
}
