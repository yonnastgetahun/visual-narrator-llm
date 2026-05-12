"use client";

import { useEffect, useRef, useState } from "react";
import { Play, Stop } from "@phosphor-icons/react";

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

  if (narrations.length === 0) {
    return (
      <div className="border border-dashed border-vn-ash bg-vn-carbon px-5 py-6 text-sm text-vn-mist">
        No narration gaps were detected for this clip, so there is no audio track to preview.
      </div>
    );
  }

  return (
    <div className="divide-y divide-vn-ash">
      {narrations.map((entry, index) => (
        <article key={`${entry.srt_index}-${entry.start_sec}`} className="border-b border-vn-ash py-6 first:pt-0 last:pb-0">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div className="flex-1">
              <p className="font-mono text-xs text-vn-amber tracking-[0.18em] uppercase">
                {formatClock(entry.start_sec)} → {formatClock(entry.end_sec)} · {formatDuration(entry.gap_duration_sec)} · {entry.gap_type}
              </p>
              <p className="mt-3 text-base leading-7 text-vn-fog">{entry.description}</p>
            </div>
            <button
              className={`inline-flex h-10 w-10 flex-none items-center justify-center rounded-full border transition-colors ${
                playingIndex === index
                  ? "border-vn-amber bg-vn-amber text-vn-black"
                  : "border-vn-ash bg-vn-carbon text-vn-mist hover:border-vn-amber hover:text-vn-amber"
              }`}
              onClick={() => togglePlayback(entry, index)}
              type="button"
              aria-label={playingIndex === index ? "Stop playback" : "Play narration"}
            >
              {playingIndex === index ? (
                <Stop size={14} weight="fill" />
              ) : (
                <Play size={14} weight="fill" />
              )}
            </button>
          </div>
        </article>
      ))}
    </div>
  );
}
