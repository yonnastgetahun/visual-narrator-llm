"use client";

import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { useState, useRef } from "react";

const EASE_VN = [0.16, 1, 0.3, 1] as const;
const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://visual-narrator-llm-production.up.railway.app";

type DownloadCard = {
  key: string;
  label: string;
  description: string;
  ext: string;
  size: string;
  icon: React.ReactNode;
};

function filenameFromDisposition(disposition: string | null, fallback: string) {
  if (!disposition) return fallback;
  const match = disposition.match(/filename="?([^"]+)"?/i);
  return match?.[1] ?? fallback;
}

function triggerDownload(blob: Blob, filename: string) {
  const objectUrl = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(objectUrl);
}

function Mp3Icon() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <path d="M11 3v10m0 0a3 3 0 1 0 3 3V6.5l4-1.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
function SrtIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <rect x="3" y="7" width="16" height="3" rx="1" stroke="currentColor" strokeWidth="1.4" />
      <rect x="3" y="13" width="10" height="3" rx="1" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}
function PdfIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <path d="M6 2h8l4 4v14a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
      <path d="M14 2v4h4M8 13h6M8 16h4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

// Waveform playback visualizer
function PlaybackWaveform({ playing }: { playing: boolean }) {
  const bars = 32;
  return (
    <div className="flex items-center gap-[2px] h-8" aria-hidden="true">
      {Array.from({ length: bars }, (_, i) => {
        const h = 20 + Math.sin(i * 0.7) * 14 + Math.sin(i * 1.3) * 8;
        return (
          <motion.div
            key={i}
            className="flex-none w-[3px] rounded-full bg-vn-amber origin-center"
            style={{ height: `${h}%` }}
            animate={
              playing
                ? {
                    scaleY: [0.3, 1, 0.4, 0.9, 0.3],
                    opacity: [0.5, 1, 0.6, 1, 0.5],
                  }
                : { scaleY: 0.3, opacity: 0.3 }
            }
            transition={
              playing
                ? { duration: 1.1 + i * 0.04, repeat: Infinity, ease: "easeInOut", delay: i * 0.03 }
                : { duration: 0.4 }
            }
          />
        );
      })}
    </div>
  );
}

// Compliance grade badge
function GradeBadge({ grade }: { grade: string }) {
  const isGood = ["A+", "A", "A-", "B+", "B"].includes(grade);
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 200, damping: 18, delay: 0.7 }}
      className={`flex flex-col items-center justify-center border w-20 h-20 ${
        isGood
          ? "border-vn-amber-border bg-vn-amber-glow text-vn-amber"
          : "border-vn-line bg-vn-carbon text-vn-mist"
      }`}
    >
      <span className="font-display text-3xl font-bold leading-none">{grade}</span>
      <span className="vn-label mt-1" style={{ fontSize: "0.5625rem" }}>WCAG 2.1</span>
    </motion.div>
  );
}

export function ResultsScreen({ jobId }: { jobId: string }) {
  const [playing, setPlaying] = useState(false);
  const [downloadingKey, setDownloadingKey] = useState<string | null>(null);
  const [downloadedKey, setDownloadedKey] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);

  const downloads: DownloadCard[] = [
    {
      key: "mp3",
      label: "Audio description track",
      description: "Broadcast-ready MP3, mixed onto the source audio at each gap timestamp",
      ext: "MP3",
      size: "14.2 MB",
      icon: <Mp3Icon />,
    },
    {
      key: "srt",
      label: "SRT subtitle file",
      description: "Timed captions synced to narration gaps, UTF-8 encoded",
      ext: "SRT",
      size: "28 KB",
      icon: <SrtIcon />,
    },
    {
      key: "pdf",
      label: "Compliance report",
      description: "WCAG 2.1 AA audit with per-gap scoring, currently downloaded as JSON",
      ext: "PDF",
      size: "190 KB",
      icon: <PdfIcon />,
    },
  ];

  async function handleDownload(key: string) {
    const endpoints: Record<string, { path: string; fallbackName: string }> = {
      mp3: { path: "audio", fallbackName: `${jobId}-ad-track.mp3` },
      srt: { path: "srt", fallbackName: `${jobId}-narration.srt` },
      pdf: { path: "report", fallbackName: `${jobId}-compliance-report.json` },
    };
    const endpoint = endpoints[key];
    if (!endpoint) return;

    setDownloadingKey(key);
    try {
      const response = await fetch(`${API_URL}/api/download/${jobId}/${endpoint.path}`);
      if (!response.ok) {
        throw new Error(`Download failed with status ${response.status}`);
      }

      if (key === "pdf") {
        const payload = await response.json();
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
        triggerDownload(blob, endpoint.fallbackName);
      } else {
        const blob = await response.blob();
        const filename = filenameFromDisposition(
          response.headers.get("content-disposition"),
          endpoint.fallbackName,
        );
        triggerDownload(blob, filename);
      }

      setDownloadedKey(key);
      setTimeout(() => setDownloadedKey(null), 2200);
    } finally {
      setDownloadingKey(null);
    }
  }

  return (
    <div className="relative min-h-[100dvh] bg-vn-black px-5 py-20 flex flex-col items-center justify-center">
      {/* Letterbox */}
      <div aria-hidden="true" className="absolute left-0 right-0 top-0 h-14 bg-vn-black" />
      <div aria-hidden="true" className="absolute bottom-0 left-0 right-0 h-14 bg-vn-black" />

      <div className="w-full max-w-[620px]">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.65, ease: EASE_VN }}
          className="flex items-start justify-between gap-4"
        >
          <div>
            <span className="vn-label text-vn-amber flex items-center gap-2.5 mb-4">
              <span className="vn-amber-rule" />
              Complete
            </span>
            <h1 className="vn-title text-vn-cream">Your AD track is ready.</h1>
            <p className="mt-2 text-[0.9375rem] text-vn-mist">Job {jobId} · 47 gaps described · 88 min total</p>
          </div>
          <GradeBadge grade="A" />
        </motion.div>

        {/* Playback preview */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: EASE_VN, delay: 0.18 }}
          className="mt-8 border border-vn-line px-5 py-5"
        >
          <div className="flex items-center gap-5">
            {/* Play / pause */}
            <motion.button
              type="button"
              onClick={() => setPlaying((p) => !p)}
              whileTap={{ scale: 0.94 }}
              className="flex-none h-11 w-11 bg-vn-amber flex items-center justify-center text-vn-black hover:bg-amber-400 transition-colors duration-150"
              aria-label={playing ? "Pause preview" : "Play preview"}
            >
              {playing ? (
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                  <rect x="2" y="2" width="4" height="10" rx="1" fill="currentColor" />
                  <rect x="8" y="2" width="4" height="10" rx="1" fill="currentColor" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                  <path d="M3 2l9 5-9 5V2z" fill="currentColor" />
                </svg>
              )}
            </motion.button>
            <div className="flex-1 overflow-hidden">
              <PlaybackWaveform playing={playing} />
            </div>
            <span className="flex-none font-mono text-[0.8125rem] text-vn-dim">04:22</span>
          </div>
          <p className="mt-3 vn-label text-vn-dim">
            Preview · First 5 minutes of the audio description track
          </p>
        </motion.div>

        {/* Download cards */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-5 border border-vn-line divide-y divide-vn-line"
        >
          {downloads.map((card, i) => (
            <motion.div
              key={card.key}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.45, ease: EASE_VN, delay: 0.35 + i * 0.08 }}
              className="flex items-center gap-4 px-5 py-4"
            >
              {/* Icon */}
              <div className="flex-none w-9 h-9 bg-vn-carbon border border-vn-line flex items-center justify-center text-vn-mist">
                {card.icon}
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <p className="text-[0.9375rem] font-medium text-vn-cream">{card.label}</p>
                <p className="vn-label text-vn-dim mt-0.5">{card.description}</p>
              </div>

              {/* Size + download */}
              <div className="flex-none flex items-center gap-3">
                <span className="font-mono text-[0.8125rem] text-vn-dim hidden sm:block">{card.size}</span>
                <motion.button
                  type="button"
                  onClick={() => handleDownload(card.key)}
                  whileTap={{ scale: 0.96 }}
                  disabled={downloadingKey === card.key}
                  className={`px-3.5 py-1.5 text-[0.8125rem] font-medium border transition-colors duration-150 ${
                    downloadedKey === card.key
                      ? "border-vn-amber bg-vn-amber text-vn-black"
                      : "border-vn-line text-vn-mist hover:border-vn-amber hover:text-vn-amber"
                  } ${downloadingKey === card.key ? "cursor-wait opacity-70" : ""}`}
                  aria-label={`Download ${card.label}`}
                >
                  <AnimatePresence mode="wait">
                    {downloadedKey === card.key ? (
                      <motion.span
                        key="done"
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 4 }}
                        className="flex items-center gap-1.5"
                      >
                        <svg width="11" height="9" viewBox="0 0 11 9" fill="none" aria-hidden="true">
                          <path d="M1 4.5l3.5 3.5L10 1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        Saved
                      </motion.span>
                    ) : (
                      <motion.span
                        key="dl"
                        initial={{ opacity: 0, y: 4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -4 }}
                      >
                        {downloadingKey === card.key ? "..." : card.ext}
                      </motion.span>
                    )}
                  </AnimatePresence>
                </motion.button>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Score summary */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.62, duration: 0.5 }}
          className="mt-5 border border-vn-line px-5 py-4 flex flex-wrap gap-5"
        >
          {[
            { label: "Accuracy",     val: "91.4%" },
            { label: "Relevance",    val: "88.7%" },
            { label: "WCAG AA",      val: "94.2%" },
            { label: "Conciseness",  val: "86.1%" },
            { label: "Within limit", val: "47 / 47" },
          ].map((s) => (
            <div key={s.label} className="flex flex-col gap-0.5">
              <span className="font-mono text-lg text-vn-cream font-medium">{s.val}</span>
              <span className="vn-label text-vn-dim">{s.label}</span>
            </div>
          ))}
        </motion.div>

        {/* Cost row */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.72 }}
          className="mt-3 flex items-center justify-between text-[0.8125rem] font-mono"
        >
          <span className="text-vn-dim">GPT $0.083 · TTS $0.041 · Total cost to generate</span>
          <span className="text-vn-mist">$0.124</span>
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.5, ease: EASE_VN }}
          className="mt-10 flex flex-wrap items-center gap-4"
        >
          <Link href="/upload">
            <motion.span
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.98 }}
              className="inline-flex cursor-pointer items-center gap-2 bg-vn-amber px-6 py-3 text-[0.9375rem] font-medium text-vn-black hover:bg-amber-400 transition-colors duration-200"
            >
              Describe another film
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
                <path d="M1.5 6.5h10M7.5 2.5l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </motion.span>
          </Link>
          <button
            type="button"
            className="text-[0.9375rem] text-vn-mist underline underline-offset-4 decoration-vn-ash hover:text-vn-fog transition-colors duration-200"
          >
            Share results
          </button>
        </motion.div>
      </div>
    </div>
  );
}
