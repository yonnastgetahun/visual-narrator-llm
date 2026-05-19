"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import { useState, useEffect, useRef } from "react";

const EASE_VN = [0.16, 1, 0.3, 1] as const;
const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "https://api.vnpoverview.com").trim();

type StepKey = "downloading" | "detecting" | "describing" | "generating" | "mixing";

type Step = {
  key: StepKey;
  label: string;
  detail: string;
  durationMs: number;
};

const STEPS: Step[] = [
  { key: "downloading", label: "Downloading footage",      detail: "Pulling source from storage",       durationMs: 3200  },
  { key: "detecting",   label: "Detecting narration gaps", detail: "Analysing dialogue and music cues", durationMs: 4800  },
  { key: "describing",  label: "Describing frames",        detail: "GPT-4o Vision — analysing gaps",    durationMs: 7600  },
  { key: "generating",  label: "Generating audio",         detail: "TTS synthesis in progress",         durationMs: 5200  },
  { key: "mixing",      label: "Mixing and exporting",     detail: "Rendering final AD track",          durationMs: 3000  },
];

// Animated waveform bars — isolated for perf
function WaveformBars({ active }: { active: boolean }) {
  return (
    <div className="flex items-center gap-[3px] h-5" aria-hidden="true">
      {Array.from({ length: 12 }, (_, i) => (
        <motion.span
          key={i}
          className="w-[3px] rounded-full bg-vn-amber origin-bottom"
          animate={
            active
              ? { scaleY: [0.2, 1, 0.3, 0.8, 0.2], opacity: [0.4, 1, 0.5, 0.9, 0.4] }
              : { scaleY: 0.2, opacity: 0.2 }
          }
          transition={
            active
              ? {
                  duration: 1.2 + i * 0.07,
                  repeat: Infinity,
                  ease: "easeInOut",
                  delay: i * 0.06,
                }
              : { duration: 0.3 }
          }
          style={{ height: "20px", display: "inline-block" }}
        />
      ))}
    </div>
  );
}

// Single step row
function StepRow({
  step,
  status,
  progress,
  index,
}: {
  step: Step;
  status: "pending" | "active" | "done";
  progress?: number;
  index: number;
}) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: EASE_VN, delay: index * 0.06 }}
      className={`flex items-center gap-5 px-5 py-4 border-b border-vn-line last:border-b-0 ${
        status === "active" ? "bg-vn-carbon" : ""
      }`}
    >
      {/* Status indicator */}
      <div className="flex-none w-6 flex justify-center">
        {status === "done" && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 280, damping: 20 }}
            className="h-5 w-5 rounded-full bg-vn-amber flex items-center justify-center"
          >
            <svg width="10" height="8" viewBox="0 0 10 8" fill="none" aria-hidden="true">
              <path d="M1 4l3 3 5-6" stroke="#0A0A0A" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </motion.div>
        )}
        {status === "active" && (
          <motion.div
            animate={{ opacity: [1, 0.4, 1] }}
            transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
            className="h-2 w-2 rounded-full bg-vn-amber"
          />
        )}
        {status === "pending" && (
          <div className="h-2 w-2 rounded-full bg-vn-ash" />
        )}
      </div>

      {/* Label + detail */}
      <div className="flex-1 min-w-0">
        <p
          className={`text-[0.9375rem] font-medium ${
            status === "done" ? "text-vn-cream" : status === "active" ? "text-vn-cream" : "text-vn-dim"
          }`}
        >
          {step.label}
        </p>
        {status !== "pending" && (
          <motion.p
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="vn-label text-vn-dim mt-0.5"
          >
            {step.detail}
          </motion.p>
        )}
        {/* Progress bar for active steps with progress */}
        {status === "active" && progress !== undefined && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-2 h-[2px] bg-vn-ash overflow-hidden"
          >
            <motion.div
              className="h-full bg-vn-amber"
              initial={{ width: "0%" }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            />
          </motion.div>
        )}
      </div>

      {/* Waveform for active audio step */}
      {status === "active" && (step.key === "generating" || step.key === "mixing") && (
        <WaveformBars active />
      )}

      {/* Duration done */}
      {status === "done" && (
        <span className="flex-none font-mono text-[0.8125rem] text-vn-dim">done</span>
      )}
    </motion.div>
  );
}

// Cinematic film counter
function FilmCounter({ currentStep }: { currentStep: number }) {
  const total = STEPS.length;
  return (
    <div className="flex items-center gap-3" aria-label={`Step ${currentStep + 1} of ${total}`}>
      {STEPS.map((_, i) => (
        <motion.div
          key={i}
          className="h-[3px] flex-1"
          animate={{
            backgroundColor:
              i < currentStep
                ? "#F59E0B"
                : i === currentStep
                ? "rgba(245,158,11,0.5)"
                : "rgba(255,255,255,0.07)",
          }}
          transition={{ duration: 0.4 }}
        />
      ))}
    </div>
  );
}

export function ProcessingScreen({
  estimatedMinutes,
  jobId,
  s3Key,
  source,
}: {
  estimatedMinutes: number;
  jobId: string;
  s3Key?: string;
  source?: string;
}) {
  const router = useRouter();
  const [activeStepIdx, setActiveStepIdx] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());
  const [progress, setProgress] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const startRef = useRef(Date.now());
  const completionStartedRef = useRef(false);
  const esRef = useRef<EventSource | null>(null);

  const useAppBox = Boolean(source || s3Key);

  // App Box SSE-driven pipeline
  useEffect(() => {
    if (!useAppBox) {
      setProcessingError("No video source was provided. Go back and paste a URL or upload a file.");
      return;
    }

    const params = new URLSearchParams({ min_gap: "2.0" });
    if (s3Key) params.set("s3_key", s3Key);
    else if (source) params.set("source", source);

    const es = new EventSource(`${API_URL}/api/ad?${params}`);
    esRef.current = es;

    // Track current step to avoid going backwards
    let currentIdx = -1;

    function advanceTo(idx: number) {
      if (idx <= currentIdx) return;
      currentIdx = idx;
      setActiveStepIdx(idx);
      setProgress(0);
      setCompletedSteps((prev) => {
        const next = new Set(prev);
        for (let i = 0; i < idx; i++) next.add(i);
        return next;
      });
    }

    es.addEventListener("step", (event: MessageEvent) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const data = JSON.parse(event.data) as Record<string, any>;
      switch (data.step) {
        case "download":
          advanceTo(0);
          break;
        case "gaps":
          advanceTo(1);
          break;
        case "describing":
        case "skipping":
          advanceTo(2);
          if (data.current && data.total) {
            setProgress(Math.round((data.current / data.total) * 100));
          }
          break;
        case "synthesizing":
          advanceTo(3);
          if (data.current && data.total) {
            setProgress(Math.round((data.current / data.total) * 100));
          }
          break;
      }
    });

    es.addEventListener("complete", (event: MessageEvent) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const data = JSON.parse(event.data) as { manifest?: Record<string, any> } | Record<string, any>;
      const manifest = "manifest" in data && data.manifest ? data.manifest : data;
      es.close();
      esRef.current = null;

      setCompletedSteps(new Set([0, 1, 2, 3, 4]));
      setActiveStepIdx(STEPS.length);

      if (completionStartedRef.current) return;
      completionStartedRef.current = true;

      const resultJobId = String(manifest.job_id || jobId);
      const resultParams = new URLSearchParams();
      if (manifest.gaps_found != null) resultParams.set("gaps", String(manifest.gaps_found));
      if (manifest.duration_seconds != null) resultParams.set("duration", String(manifest.duration_seconds));
      if (manifest.gpt_cost_estimate != null) resultParams.set("gpt", String(manifest.gpt_cost_estimate));
      if (manifest.tts_cost_estimate != null) resultParams.set("tts", String(manifest.tts_cost_estimate));
      if (manifest.total_cost_estimate != null) resultParams.set("total", String(manifest.total_cost_estimate));
      if (manifest.compliance_level != null) resultParams.set("wcag", String(manifest.compliance_level));

      void fetch(`/api/jobs/${jobId}/complete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ estimatedMinutes }),
      }).finally(() => {
        const suffix = resultParams.size ? `?${resultParams.toString()}` : "";
        setTimeout(() => router.push(`/results/${resultJobId}${suffix}`), 800);
      });
    });

    es.addEventListener("error", (event: MessageEvent) => {
      let message = "Processing failed. Try another URL or direct video file.";
      try {
        const data = JSON.parse(event.data) as { message?: string };
        if (data.message) message = data.message;
      } catch {
        // Keep fallback message.
      }
      es.close();
      esRef.current = null;
      setProcessingError(message);
    });

    es.onerror = () => {
      es.close();
      esRef.current = null;
      setProcessingError("Processing disconnected before completion. Please go back and try another URL.");
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [estimatedMinutes, jobId, router, s3Key, source, useAppBox]);

  // Legacy simulated pipeline is intentionally disabled. Real-output wiring
  // requires a source URL or uploaded S3 key so downloads point at App Box outputs.
  useEffect(() => {
    if (useAppBox) return;
    setProcessingError("No video source was provided. Go back and paste a URL or upload a file.");
  }, [useAppBox]);

  // Elapsed timer
  useEffect(() => {
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 1000);
    return () => clearInterval(t);
  }, []);

  const totalSecs = STEPS.reduce((s, st) => s + st.durationMs, 0) / 1000;
  const remaining = Math.max(0, Math.round(totalSecs - elapsed));
  const remainMins = Math.floor(remaining / 60);
  const remainSecs = remaining % 60;

  return (
    <div className="relative min-h-[100dvh] bg-vn-black flex flex-col items-center justify-center px-5 py-20">
      {/* Letterbox bars */}
      <div aria-hidden="true" className="absolute left-0 right-0 top-0 h-14 bg-vn-black" />
      <div aria-hidden="true" className="absolute bottom-0 left-0 right-0 h-14 bg-vn-black" />

      <div className="w-full max-w-[580px]">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: EASE_VN }}
          className="mb-8"
        >
          <span className="vn-label text-vn-amber flex items-center gap-2.5 mb-4">
            <span className="vn-amber-rule" />
            Processing
          </span>
          <div className="flex items-end justify-between gap-4">
            <div>
              <h1 className="vn-title text-vn-cream leading-none">Describing your film</h1>
              <p className="mt-2 text-[0.9375rem] text-vn-dim font-mono"># {jobId}</p>
            </div>
            {/* Time remaining */}
            <div className="flex-none text-right">
              <motion.span
                key={remaining}
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                className="font-mono text-2xl text-vn-cream font-medium block"
              >
                {String(remainMins).padStart(2, "0")}:{String(remainSecs).padStart(2, "0")}
              </motion.span>
              <span className="vn-label text-vn-dim">remaining</span>
            </div>
          </div>
        </motion.div>

        {/* Progress track */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="mb-6"
        >
          <FilmCounter currentStep={activeStepIdx} />
        </motion.div>

        {/* Steps */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: EASE_VN, delay: 0.28 }}
          className="border border-vn-line"
        >
          <AnimatePresence initial={false}>
            {STEPS.map((step, i) => (
              <StepRow
                key={step.key}
                step={step}
                status={
                  completedSteps.has(i)
                    ? "done"
                    : i === activeStepIdx
                    ? "active"
                    : "pending"
                }
                progress={i === activeStepIdx ? progress : undefined}
                index={i}
              />
            ))}
          </AnimatePresence>
        </motion.div>

        {/* Error state */}
        {processingError && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 border border-red-400/30 bg-red-400/5 px-4 py-3"
          >
            <p className="text-[0.8125rem] text-red-400">{processingError}</p>
            <p className="mt-2 text-[0.75rem] text-vn-dim">
              YouTube may occasionally block server-side downloads. Direct video files are the most reliable input.
            </p>
            <a href="/upload" className="mt-2 block vn-label text-vn-dim hover:text-vn-mist transition-colors">
              ← Go back
            </a>
          </motion.div>
        )}

        {/* Bottom note */}
        {!processingError && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="mt-6 text-center text-[0.8125rem] text-vn-dim leading-relaxed"
          >
            You can close this tab. We will email you when the track is ready.
            <br />
            <span className="text-vn-amber">Job ID saved to clipboard.</span>
          </motion.p>
        )}
      </div>
    </div>
  );
}
