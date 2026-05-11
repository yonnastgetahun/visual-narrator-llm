"use client";

import { useRef, useState } from "react";

import AgeGate from "@/components/AgeGate";
import { ProgressStream } from "@/components/ProgressStream";
import { ResultsPanel } from "@/components/ResultsPanel";
import { UrlInput } from "@/components/UrlInput";
import type { Manifest, StepEvent } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const YOUTUBE_URL_PATTERN = /^https?:\/\/((www|m)\.)?(youtube\.com|youtu\.be)(\/|$)/i;

const isYouTubeUrl = (url: string) => YOUTUBE_URL_PATTERN.test(url);

export default function Page() {
  const eventSourceRef = useRef<EventSource | null>(null);

  const [gateConfirmed, setGateConfirmed] = useState(false);
  const [url, setUrl] = useState("");
  const [steps, setSteps] = useState<StepEvent[]>([]);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const showDownloadHint = steps[steps.length - 1]?.step === "download";

  function resetRunState() {
    setSteps([]);
    setManifest(null);
    setError(null);
  }

  function cancelActiveRun() {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setProcessing(false);
  }

  function handleSubmit(nextUrl: string) {
    cancelActiveRun();
    resetRunState();
    setProcessing(true);

    const endpoint = `${API_BASE}/api/ad-adult?source=${encodeURIComponent(nextUrl)}&min_gap=2.0`;
    const eventSource = new EventSource(endpoint);
    eventSourceRef.current = eventSource;

    eventSource.addEventListener("step", (event) => {
      const data = JSON.parse((event as MessageEvent).data) as StepEvent;
      setSteps((previous) => [...previous, data]);
    });

    eventSource.addEventListener("complete", (event) => {
      const data = JSON.parse((event as MessageEvent).data) as { manifest: Manifest };
      setManifest(data.manifest);
      setProcessing(false);
      eventSource.close();
      eventSourceRef.current = null;
    });

    eventSource.addEventListener("error", (event) => {
      const maybeData = "data" in event ? (event as MessageEvent).data : "";
      if (maybeData) {
        const data = JSON.parse(maybeData) as { message: string };
        setError(data.message);
      } else {
        setError("Connection to the demo API was interrupted.");
      }
      setProcessing(false);
      eventSource.close();
      eventSourceRef.current = null;
    });
  }

  if (!gateConfirmed) {
    return <AgeGate onConfirm={() => setGateConfirmed(true)} />;
  }

  return (
    <main className="relative overflow-hidden">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-10 px-6 py-12 md:px-10">
        <section className="rounded-[2.5rem] border border-white/10 bg-slate-950/55 p-8 shadow-2xl shadow-cyan-950/20 backdrop-blur md:p-12">
          <p className="text-xs uppercase tracking-[0.4em] text-cyan-300">Visual Narrator</p>
          <h1 className="mt-5 max-w-3xl text-4xl font-semibold leading-tight text-white md:text-6xl">
            Adult Content Audio Description Demo
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-300">
            Submit a direct MP4 or hosted video URL to generate an audio description track for professional
            accessibility review.
          </p>
          <div className="mt-8">
            <UrlInput
              value={url}
              processing={processing}
              onChange={setUrl}
              onSubmit={(nextUrl) => {
                setUrl(nextUrl);
                handleSubmit(nextUrl);
              }}
            />
          </div>
          {isYouTubeUrl(url) ? (
            <div className="mt-6 rounded-2xl border border-amber-400/30 bg-amber-400/10 px-4 py-4 text-sm text-amber-100">
              Hosted direct video URLs are recommended for the adult demo pipeline.
            </div>
          ) : null}
          {error ? (
            <div className="mt-6 rounded-2xl border border-rose-400/30 bg-rose-400/10 px-4 py-4 text-sm text-rose-100">
              {error}
            </div>
          ) : null}
        </section>

        {processing ? (
          <ProgressStream
            isComplete={Boolean(manifest)}
            onCancel={cancelActiveRun}
            showDownloadHint={showDownloadHint}
            steps={steps}
          />
        ) : null}

        {manifest ? <ResultsPanel manifest={manifest} /> : null}

        <footer className="mx-auto mt-8 max-w-lg text-center text-xs text-zinc-500">
          This demo collects nothing. No analytics, no cookies, no session data. The URL you submit is sent to our
          audio description service and immediately discarded. It is never stored, logged, or associated with any
          identifier.
        </footer>
      </div>
    </main>
  );
}
