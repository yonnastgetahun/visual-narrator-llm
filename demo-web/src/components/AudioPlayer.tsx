"use client";

import { useEffect, useRef, useState } from "react";

import { formatClock, formatDuration } from "@/lib/format";
import type { NarrationEntry } from "@/lib/types";

type AudioPlayerProps = {
  narrations: NarrationEntry[];
};

export function AudioPlayer({ narrations }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingIndex, setPlayingIndex] = useState<number | null>(null);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);

  function togglePlayback(entry: NarrationEntry, index: number) {
    if (audioRef.current && playingIndex === index) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null;
      setPlayingIndex(null);
      return;
    }

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    const audio = new Audio(`data:${entry.audio_mime};base64,${entry.audio_data}`);
    audio.onended = () => {
      setPlayingIndex(null);
      audioRef.current = null;
    };
    audioRef.current = audio;
    setPlayingIndex(index);
    void audio.play().catch(() => {
      setPlayingIndex(null);
      audioRef.current = null;
    });
  }

  return (
    <div className="space-y-4">
      {narrations.map((entry, index) => (
        <article key={`${entry.srt_index}-${entry.start_sec}`} className="rounded-2xl border border-white/10 bg-white/5 p-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
            <div>
              <p className="font-mono text-xs uppercase tracking-[0.22em] text-cyan-200">
                {formatClock(entry.start_sec)} → {formatClock(entry.end_sec)} · {formatDuration(entry.gap_duration_sec)} · {entry.gap_type}
              </p>
              <p className="mt-3 text-base leading-7 text-slate-100">{entry.description}</p>
            </div>
            <button
              className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-cyan-300/40 bg-cyan-300/10 text-lg text-cyan-100 transition hover:bg-cyan-300/20"
              onClick={() => togglePlayback(entry, index)}
              type="button"
            >
              {playingIndex === index ? "■" : "▶"}
            </button>
          </div>
        </article>
      ))}
    </div>
  );
}
