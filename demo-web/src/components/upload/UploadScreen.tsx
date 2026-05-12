"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import { useState, useCallback, useRef } from "react";

const EASE_VN = [0.16, 1, 0.3, 1] as const;

type InputMode = "url" | "file";
type FileSizeInfo = { name: string; size: string; duration: string; cost: string };

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Rough estimate: 1GB ~ 90 min film
function estimateDuration(bytes: number): string {
  const mins = Math.round((bytes / (1024 * 1024 * 1024)) * 90 * 60);
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  if (h === 0) return `~${m}m`;
  return `~${h}h ${m}m`;
}

export function UploadScreen() {
  const router = useRouter();
  const [mode, setMode] = useState<InputMode>("url");
  const [url, setUrl] = useState("");
  const [fileInfo, setFileInfo] = useState<FileSizeInfo | null>(null);
  const [dragging, setDragging] = useState(false);
  const [urlError, setUrlError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    const durationStr = estimateDuration(file.size);
    const mins = Math.round((file.size / (1024 * 1024 * 1024)) * 90);
    const cost = `$${Math.max(39, Math.round(mins / 90 * 39)).toFixed(0)}`;
    setFileInfo({
      name: file.name,
      size: formatBytes(file.size),
      duration: durationStr,
      cost,
    });
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file && file.type.startsWith("video/")) handleFile(file);
    },
    [handleFile]
  );

  const validateUrl = (val: string) => {
    try {
      new URL(val);
      return true;
    } catch {
      return false;
    }
  };

  function handleSubmit() {
    if (mode === "url") {
      if (!url.trim() || !validateUrl(url.trim())) {
        setUrlError("Enter a valid video URL — YouTube, Vimeo, or direct link.");
        return;
      }
    }
    setSubmitting(true);
    // In real impl: POST to API, get jobId back, then navigate
    setTimeout(() => {
      router.push("/processing/demo-job-001");
    }, 800);
  }

  return (
    <div className="relative min-h-[100dvh] bg-vn-black px-5 pt-20 pb-16 flex flex-col items-center justify-center">
      {/* Back nav */}
      <motion.a
        href="/"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="absolute left-6 top-7 flex items-center gap-2 text-vn-dim hover:text-vn-mist transition-colors vn-label"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
          <path d="M10 7H2m4-4L2 7l4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        Back
      </motion.a>

      <div className="w-full max-w-[640px]">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: EASE_VN }}
        >
          <span className="vn-label text-vn-amber flex items-center gap-2.5">
            <span className="vn-amber-rule" />
            Visual Narrator
          </span>
          <h1 className="vn-title mt-4 text-vn-cream">Describe a film</h1>
          <p className="mt-3 text-[0.9375rem] leading-relaxed text-vn-mist max-w-[46ch]">
            Upload a video file or paste a URL. We handle the rest.
          </p>
        </motion.div>

        {/* Mode toggle */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: EASE_VN, delay: 0.15 }}
          className="mt-10 flex gap-0 border border-vn-line"
        >
          {(["url", "file"] as InputMode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`relative flex-1 py-2.5 text-[0.8125rem] font-medium tracking-wide transition-colors duration-200 ${
                mode === m ? "bg-vn-carbon text-vn-cream" : "text-vn-dim hover:text-vn-mist"
              }`}
            >
              {m === "url" ? "Paste URL" : "Upload file"}
              {mode === m && (
                <motion.span
                  layoutId="mode-indicator"
                  className="absolute bottom-0 left-0 right-0 h-[2px] bg-vn-amber"
                />
              )}
            </button>
          ))}
        </motion.div>

        {/* Input area */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: EASE_VN, delay: 0.25 }}
          className="mt-4"
        >
          <AnimatePresence mode="wait">
            {mode === "url" ? (
              <motion.div
                key="url"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.28, ease: EASE_VN }}
              >
                <label htmlFor="video-url" className="vn-label text-vn-dim mb-2 block">
                  YouTube, Vimeo, or direct video URL
                </label>
                <input
                  id="video-url"
                  type="url"
                  value={url}
                  onChange={(e) => {
                    setUrl(e.target.value);
                    setUrlError(null);
                  }}
                  placeholder="https://youtube.com/watch?v=..."
                  className="w-full bg-vn-carbon border border-vn-line px-4 py-3.5 text-[0.9375rem] text-vn-cream placeholder:text-vn-dim outline-none focus:border-vn-amber-border transition-colors duration-200"
                />
                {urlError && (
                  <motion.p
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-2 text-[0.8125rem] text-red-400"
                  >
                    {urlError}
                  </motion.p>
                )}
              </motion.div>
            ) : (
              <motion.div
                key="file"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.28, ease: EASE_VN }}
              >
                <input
                  ref={fileRef}
                  type="file"
                  accept="video/*"
                  className="sr-only"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFile(file);
                  }}
                />
                <button
                  type="button"
                  onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={onDrop}
                  onClick={() => fileRef.current?.click()}
                  className={`w-full border border-dashed py-14 flex flex-col items-center gap-3 transition-colors duration-200 ${
                    dragging
                      ? "border-vn-amber bg-vn-amber-glow"
                      : "border-vn-ash hover:border-vn-dim bg-vn-carbon"
                  }`}
                >
                  {/* Film reel icon */}
                  <svg width="32" height="32" viewBox="0 0 32 32" fill="none" aria-hidden="true" className="text-vn-dim">
                    <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="1.5" />
                    <circle cx="16" cy="16" r="5" stroke="currentColor" strokeWidth="1.5" />
                    <circle cx="16" cy="6"  r="2.5" stroke="currentColor" strokeWidth="1.5" />
                    <circle cx="16" cy="26" r="2.5" stroke="currentColor" strokeWidth="1.5" />
                    <circle cx="6"  cy="16" r="2.5" stroke="currentColor" strokeWidth="1.5" />
                    <circle cx="26" cy="16" r="2.5" stroke="currentColor" strokeWidth="1.5" />
                  </svg>
                  <span className="text-[0.9375rem] text-vn-mist">
                    {dragging ? "Release to add film" : "Drop a video file here"}
                  </span>
                  <span className="vn-label text-vn-dim">MP4, MOV, MKV, AVI · up to 4 GB</span>
                </button>

                {/* File preview */}
                <AnimatePresence>
                  {fileInfo && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-3 border border-vn-amber-border bg-vn-amber-glow px-4 py-3 flex items-center justify-between"
                    >
                      <div className="flex flex-col gap-0.5">
                        <span className="text-[0.9375rem] text-vn-cream font-medium truncate max-w-[24ch]">{fileInfo.name}</span>
                        <span className="vn-label text-vn-dim">{fileInfo.size} · {fileInfo.duration} estimated</span>
                      </div>
                      <span className="font-mono text-vn-amber font-medium">{fileInfo.cost}</span>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* Pricing summary */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="mt-6 border-t border-vn-line pt-5 flex items-center justify-between"
        >
          <div className="flex flex-col gap-1">
            <span className="text-[0.9375rem] text-vn-mist">90-minute film</span>
            <span className="vn-label text-vn-dim">MP3 + SRT + compliance report</span>
          </div>
          <div className="flex flex-col items-end gap-0.5">
            <span className="font-mono text-2xl text-vn-cream font-medium">$39</span>
            <span className="vn-label text-vn-dim">flat rate</span>
          </div>
        </motion.div>

        {/* Estimated time */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.48 }}
          className="mt-3 flex items-center gap-2 text-[0.8125rem] text-vn-dim"
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
            <circle cx="6" cy="6" r="5" stroke="currentColor" strokeWidth="1" />
            <path d="M6 3.5V6l1.5 1.5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
          </svg>
          Estimated processing time: 18–22 minutes
        </motion.div>

        {/* Submit */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.55, duration: 0.5, ease: EASE_VN }}
          className="mt-8"
        >
          <motion.button
            type="button"
            onClick={handleSubmit}
            disabled={submitting}
            whileHover={{ scale: submitting ? 1 : 1.01 }}
            whileTap={{ scale: submitting ? 1 : 0.98 }}
            className="w-full bg-vn-amber py-4 text-[0.9375rem] font-medium text-vn-black tracking-wide disabled:opacity-60 transition-opacity duration-200"
          >
            {submitting ? (
              <span className="flex items-center justify-center gap-2.5">
                <span className="h-3.5 w-3.5 rounded-full border-2 border-vn-black border-t-transparent animate-spin" />
                Starting…
              </span>
            ) : (
              "Start audio description  —  $39"
            )}
          </motion.button>
          <p className="mt-3 text-center text-[0.8125rem] text-vn-dim">
            No account required. Billed once on completion.
          </p>
        </motion.div>
      </div>
    </div>
  );
}
