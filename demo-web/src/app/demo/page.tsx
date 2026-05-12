"use client";

import { useEffect, useRef, useState } from "react";
import { track } from "@vercel/analytics";

import { ProgressStream } from "@/components/ProgressStream";
import { ResultsPanel } from "@/components/ResultsPanel";
import { ScorePanel } from "@/components/ScorePanel";
import { ThemeToggle } from "@/components/ThemeToggle";
import { UrlInput } from "@/components/UrlInput";
import type { Manifest, ScoreReport, StepEvent } from "@/lib/types";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://visual-narrator-llm-production.up.railway.app";

function isYouTubeUrl(url: string) {
  return url.includes("youtube.com") || url.includes("youtu.be");
}

export default function DemoPage() {
  const [url, setUrl] = useState("");
  const [processing, setProcessing] = useState(false);
  const [steps, setSteps] = useState<StepEvent[]>([]);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scoring, setScoring] = useState(false);
  const [scoringProgress, setScoringProgress] = useState<{ current: number; total: number } | null>(null);
  const [report, setReport] = useState<ScoreReport | null>(null);
  const [isComplete, setIsComplete] = useState(false);
  const [showDownloadHint, setShowDownloadHint] = useState(false);

  const esRef = useRef<EventSource | null>(null);
  const downloadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scoringEsRef = useRef<EventSource | null>(null);

  // Track download hint: show if download step is active for >4s
  useEffect(() => {
    const latest = steps[steps.length - 1];
    if (latest?.step === "download") {
      downloadTimerRef.current = setTimeout(() => {
        setShowDownloadHint(true);
      }, 4000);
    } else {
      if (downloadTimerRef.current) {
        clearTimeout(downloadTimerRef.current);
        downloadTimerRef.current = null;
      }
      setShowDownloadHint(false);
    }
    return () => {
      if (downloadTimerRef.current) {
        clearTimeout(downloadTimerRef.current);
        downloadTimerRef.current = null;
      }
    };
  }, [steps]);

  function cancelActiveRun() {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    setProcessing(false);
    setSteps([]);
    setIsComplete(false);
    setShowDownloadHint(false);
  }

  function handleSubmit(submittedUrl: string) {
    cancelActiveRun();
    setManifest(null);
    setReport(null);
    setScoringProgress(null);
    setScoring(false);
    setSteps([]);
    setIsComplete(false);
    setProcessing(true);

    track("run_started", {
      video_type: isYouTubeUrl(submittedUrl) ? "youtube" : "direct",
    });

    const es = new EventSource(
      `${API_URL}/api/ad?source=${encodeURIComponent(submittedUrl)}&min_gap=2.0`
    );
    esRef.current = es;

    es.addEventListener("step", (event: MessageEvent) => {
      const data = JSON.parse(event.data) as StepEvent;
      setSteps((prev) => [...prev, data]);
    });

    es.addEventListener("complete", (event: MessageEvent) => {
      const data = JSON.parse(event.data) as Manifest;
      setManifest(data);
      setIsComplete(true);
      setProcessing(false);
      es.close();
      esRef.current = null;

      track("run_completed", {
        video_duration: Math.round(data.duration_seconds),
        gap_count: data.gaps_found,
        cost_estimate: data.total_cost_estimate,
      });
    });

    es.onerror = () => {
      es.close();
      esRef.current = null;
      setProcessing(false);
    };
  }

  function handleScore() {
    if (!manifest) return;
    setScoring(true);
    setScoringProgress(null);

    const scoreEs = new EventSource(
      `${API_URL}/api/score?source=${encodeURIComponent(manifest.source)}&manifest=${encodeURIComponent(JSON.stringify(manifest))}`
    );
    scoringEsRef.current = scoreEs;

    // Try SSE-style first; fall back to POST if needed
    scoreEs.addEventListener("scoring", (event: MessageEvent) => {
      const data = JSON.parse(event.data) as { current: number; total: number };
      setScoringProgress(data);
    });

    scoreEs.addEventListener("report", (event: MessageEvent) => {
      const data = JSON.parse(event.data) as ScoreReport;
      setReport(data);
      setScoring(false);
      scoreEs.close();
      scoringEsRef.current = null;
    });

    scoreEs.onerror = () => {
      scoreEs.close();
      scoringEsRef.current = null;
      // Fall back to POST
      void fetch(`${API_URL}/api/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: manifest.source, manifest }),
      })
        .then((res) => res.json())
        .then((data: ScoreReport) => {
          setReport(data);
          setScoring(false);
        })
        .catch(() => {
          setScoring(false);
        });
    };
  }

  function resetAll() {
    cancelActiveRun();
    if (scoringEsRef.current) {
      scoringEsRef.current.close();
      scoringEsRef.current = null;
    }
    setUrl("");
    setManifest(null);
    setReport(null);
    setScoringProgress(null);
    setScoring(false);
    setIsComplete(false);
  }

  return (
    <main className="min-h-dvh bg-vn-black">
      {/* Page header */}
      <header className="border-b border-vn-ash px-6 py-6 md:px-[8vw]">
        <div className="flex items-baseline justify-between">
          <span className="vn-label text-vn-amber flex items-center gap-2.5">
            <span className="vn-amber-rule" />
            Live Demo
          </span>
          <div className="flex items-center gap-4">
            <span className="vn-label text-vn-dim">demo.vnpoverview.com</span>
            <ThemeToggle />
          </div>
        </div>
        <h1 className="font-display mt-4 text-3xl text-vn-cream">
          See the pipeline run
        </h1>
        <p className="mt-2 text-sm text-vn-mist leading-relaxed max-w-prose">
          Paste a YouTube link, Vimeo URL, or direct video URL. The pipeline detects silence gaps, samples real frames, writes WCAG-compliant audio description copy, and synthesises voice — live, in your browser.
        </p>
      </header>

      <div className="px-6 py-12 md:px-[8vw]">
        <div className="max-w-[860px] space-y-10">

          {/* URL input — shown when not yet started */}
          {!processing && !manifest ? (
            <div>
              <span className="vn-label text-vn-mist flex items-center gap-2.5 mb-5">
                <span className="vn-amber-rule" />
                Source
              </span>
              <UrlInput
                value={url}
                processing={processing}
                onChange={setUrl}
                onSubmit={handleSubmit}
              />
            </div>
          ) : null}

          {/* Live pipeline progress */}
          {processing ? (
            <ProgressStream
              steps={steps}
              isComplete={isComplete}
              onCancel={cancelActiveRun}
              showDownloadHint={showDownloadHint}
            />
          ) : null}

          {/* Results — shown once manifest is ready */}
          {manifest ? (
            <>
              <ResultsPanel manifest={manifest} />
              <ScorePanel
                onScore={handleScore}
                scoring={scoring}
                scoringProgress={scoringProgress}
                report={report}
              />

              {/* Reset */}
              <div className="pt-4 border-t border-vn-ash">
                <button
                  className="vn-label text-vn-dim transition-colors hover:text-vn-mist"
                  onClick={resetAll}
                  type="button"
                >
                  Describe another film
                </button>
              </div>
            </>
          ) : null}

        </div>
      </div>
    </main>
  );
}
