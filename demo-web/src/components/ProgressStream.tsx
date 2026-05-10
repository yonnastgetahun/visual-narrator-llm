"use client";

import type { StepEvent } from "@/lib/types";

type ProgressStreamProps = {
  steps: StepEvent[];
  isComplete: boolean;
  onCancel: () => void;
};

type RowState = "pending" | "active" | "done";
type GapsDoneStep = Extract<StepEvent, { step: "gaps_done" }>;
type ProgressCountStep = Extract<StepEvent, { step: "describing" | "synthesizing" | "scoring" }>;

function latestGapsDone(steps: StepEvent[]) {
  return [...steps].reverse().find((step) => step.step === "gaps_done") as GapsDoneStep | undefined;
}

function latestProgressStep(steps: StepEvent[], name: ProgressCountStep["step"]) {
  return [...steps].reverse().find((step) => step.step === name) as ProgressCountStep | undefined;
}

function rowClass(state: RowState) {
  if (state === "done") {
    return "border-emerald-400/40 bg-emerald-400/10 text-emerald-100";
  }
  if (state === "active") {
    return "border-cyan-400/40 bg-cyan-400/10 text-cyan-50";
  }
  return "border-white/10 bg-white/5 text-slate-400";
}

function indicator(state: RowState) {
  if (state === "done") {
    return <span className="text-lg">✓</span>;
  }
  if (state === "active") {
    return <span className="h-3 w-3 animate-spin rounded-full border-2 border-cyan-200 border-t-transparent" />;
  }
  return <span className="h-3 w-3 rounded-full bg-slate-600" />;
}

export function ProgressStream({ steps, isComplete, onCancel }: ProgressStreamProps) {
  const downloadSeen = steps.some((step) => step.step === "download");
  const gapsSeen = steps.some((step) => step.step === "gaps");
  const gapsDone = latestGapsDone(steps);
  const latestDescribing = latestProgressStep(steps, "describing");
  const latestSynthesizing = latestProgressStep(steps, "synthesizing");
  const latest = steps[steps.length - 1];

  const rows = [
    {
      key: "download",
      label: "Downloading video",
      state: downloadSeen && latest?.step !== "download" ? "done" : downloadSeen ? "active" : "pending",
      detail: null,
    },
    {
      key: "gaps",
      label: "Detecting narration gaps",
      state: gapsDone ? "done" : gapsSeen ? "active" : "pending",
      detail: gapsDone ? `${gapsDone.gaps} gaps found · ${Math.round(gapsDone.duration_seconds / 60)}m ${Math.round(gapsDone.duration_seconds % 60)}s` : null,
    },
    {
      key: "describing",
      label: "Describing frames",
      state: latestSynthesizing || isComplete ? "done" : latest?.step === "describing" ? "active" : "pending",
      detail: latestDescribing ? `${latestDescribing.current} / ${latestDescribing.total}` : "— / —",
    },
    {
      key: "synthesizing",
      label: "Synthesizing audio",
      state: isComplete ? "done" : latest?.step === "synthesizing" ? "active" : "pending",
      detail: latestSynthesizing ? `${latestSynthesizing.current} / ${latestSynthesizing.total}` : "— / —",
    },
  ] as const;

  return (
    <section className="rounded-[2rem] border border-white/10 bg-slate-950/65 p-6 shadow-2xl shadow-cyan-950/20 backdrop-blur">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.32em] text-cyan-300">Live Pipeline</p>
          <h2 className="mt-2 text-2xl font-semibold text-white">Processing your source</h2>
        </div>
        <button className="text-sm text-slate-300 transition hover:text-white" onClick={onCancel} type="button">
          Cancel
        </button>
      </div>

      <div className="mt-6 grid gap-3">
        {rows.map((row) => (
          <div
            key={row.key}
            className={`flex items-center justify-between rounded-2xl border px-4 py-4 ${rowClass(row.state)}`}
          >
            <div className="flex items-center gap-3">
              {indicator(row.state)}
              <span className="font-medium">{row.label}</span>
            </div>
            <span className="text-sm">{row.detail}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
