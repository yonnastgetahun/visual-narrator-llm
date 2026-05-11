"use client";

import { track } from "@vercel/analytics";
import { useRef, useState } from "react";

import { ProgressStream } from "@/components/ProgressStream";
import { ResultsPanel } from "@/components/ResultsPanel";
import { ScorePanel } from "@/components/ScorePanel";
import { UrlInput } from "@/components/UrlInput";
import type { Manifest, ScoreReport, StepEvent } from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const YOUTUBE_URL_PATTERN = /^https?:\/\/((www|m)\.)?(youtube\.com|youtu\.be)(\/|$)/i;

type ParsedSseMessage = {
  event: string;
  data: string;
};

function parseSseBuffer(buffer: string): { messages: ParsedSseMessage[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const chunks = normalized.split("\n\n");
  const rest = normalized.endsWith("\n\n") ? "" : chunks.pop() ?? "";
  const messages = chunks
    .map((chunk) => {
      const lines = chunk.split("\n");
      let event = "message";
      const dataLines: string[] = [];
      for (const line of lines) {
        if (line.startsWith("event:")) {
          event = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        }
      }
      if (!dataLines.length) {
        return null;
      }
      return { event, data: dataLines.join("\n") };
    })
    .filter((message): message is ParsedSseMessage => message !== null);

  return { messages, rest };
}

const isYouTubeUrl = (url: string) => YOUTUBE_URL_PATTERN.test(url);

export default function Page() {
  const eventSourceRef = useRef<EventSource | null>(null);
  const scoreAbortRef = useRef<AbortController | null>(null);

  const [url, setUrl] = useState("");
  const [steps, setSteps] = useState<StepEvent[]>([]);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [report, setReport] = useState<ScoreReport | null>(null);
  const [processing, setProcessing] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [scoringProgress, setScoringProgress] = useState<{ current: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const showDownloadHint = steps[steps.length - 1]?.step === "download";

  function resetRunState() {
    setSteps([]);
    setManifest(null);
    setReport(null);
    setScoring(false);
    setScoringProgress(null);
    setError(null);
  }

  function cancelActiveRun() {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (scoreAbortRef.current) {
      scoreAbortRef.current.abort();
      scoreAbortRef.current = null;
    }
    setProcessing(false);
    setScoring(false);
  }

  function handleSubmit(nextUrl: string) {
    cancelActiveRun();
    resetRunState();
    setProcessing(true);
    track("run_started", {
      video_type: isYouTubeUrl(nextUrl) ? "youtube" : "direct",
    });

    const eventSource = new EventSource(`${API_URL}/api/ad?source=${encodeURIComponent(nextUrl)}&min_gap=2.0`);
    eventSourceRef.current = eventSource;

    eventSource.addEventListener("step", (event) => {
      const data = JSON.parse((event as MessageEvent).data) as StepEvent;
      setSteps((previous) => [...previous, data]);
    });

    eventSource.addEventListener("complete", (event) => {
      const data = JSON.parse((event as MessageEvent).data) as { manifest: Manifest };
      track("run_completed", {
        video_duration: Math.round(data.manifest.duration_seconds ?? 0),
        gap_count: data.manifest.gaps_found ?? 0,
        cost_estimate: data.manifest.total_cost_estimate ?? 0,
      });
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

  async function handleScore() {
    if (!manifest) {
      return;
    }

    setScoring(true);
    setScoringProgress(null);
    setError(null);

    const controller = new AbortController();
    scoreAbortRef.current = controller;

    try {
      const response = await fetch(`${API_URL}/api/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: manifest.source, manifest }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Score request failed with status ${response.status}.`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSseBuffer(buffer);
        buffer = parsed.rest;

        for (const message of parsed.messages) {
          const data = JSON.parse(message.data) as
            | { step: "scoring"; current: number; total: number; message: string }
            | { report: ScoreReport }
            | { message: string };

          if (message.event === "step" && "current" in data && "total" in data) {
            setScoringProgress({ current: data.current, total: data.total });
          } else if (message.event === "complete" && "report" in data) {
            setReport(data.report);
            setScoring(false);
            scoreAbortRef.current = null;
            return;
          } else if (message.event === "error" && "message" in data) {
            throw new Error(data.message);
          }
        }
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") {
        return;
      }
      const message = caught instanceof Error ? caught.message : "Unknown scoring error";
      if (message !== "The user aborted a request.") {
        setError(message);
      }
    } finally {
      setScoring(false);
      scoreAbortRef.current = null;
    }
  }

  return (
    <main className="relative overflow-hidden">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-10 px-6 py-12 md:px-10">
        <section className="rounded-[2.5rem] border border-white/10 bg-slate-950/55 p-8 shadow-2xl shadow-cyan-950/20 backdrop-blur md:p-12">
          <p className="text-xs uppercase tracking-[0.4em] text-cyan-300">Visual Narrator</p>
          <h1 className="mt-5 max-w-3xl text-4xl font-semibold leading-tight text-white md:text-6xl">
            Standard Audio Description Demo
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-300">
            Paste a YouTube, Vimeo, or direct video URL to generate a WCAG-compliant audio description track in real time.
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
          {error ? (
            <div className="mt-6 rounded-2xl border border-rose-400/30 bg-rose-400/10 px-4 py-4 text-sm text-rose-100">
              {error}
            </div>
          ) : null}
        </section>

        {processing ? (
          <ProgressStream
            isComplete={false}
            onCancel={cancelActiveRun}
            showDownloadHint={showDownloadHint}
            steps={steps}
          />
        ) : null}

        {manifest ? (
          <>
            <ResultsPanel manifest={manifest} />
            <ScorePanel onScore={handleScore} report={report} scoring={scoring} scoringProgress={scoringProgress} />
          </>
        ) : null}
      </div>
    </main>
  );
}
