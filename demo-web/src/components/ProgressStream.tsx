"use client";

import type { StepEvent } from "@/lib/types";

type ProgressStreamProps = {
  steps: StepEvent[];
  isComplete: boolean;
  onCancel: () => void;
  showDownloadHint: boolean;
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
    return "border-vn-ash bg-vn-carbon text-vn-fog";
  }
  if (state === "active") {
    return "border-vn-amber/30 bg-vn-amber/10 text-vn-cream";
  }
  return "border-vn-ash bg-vn-carbon text-vn-dim";
}

function indicator(state: RowState) {
  if (state === "done") {
    return <span className="text-vn-amber text-base font-medium">✓</span>;
  }
  if (state === "active") {
    return <span className="h-2 w-2 rounded-full bg-vn-amber animate-pulse" />;
  }
  return <span className="h-2 w-2 rounded-full bg-vn-ash" />;
}

export function ProgressStream({ steps, isComplete, onCancel, showDownloadHint }: ProgressStreamProps) {
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
    <section className="bg-vn-ink border border-vn-ash p-8">
      <div className="flex items-start justify-between mb-8">
        <div>
          <span className="vn-label text-vn-amber flex items-center gap-2.5 mb-4">
            <span className="vn-amber-rule" />
            Live Pipeline
          </span>
          <h2 className="font-display text-2xl text-vn-cream leading-tight">Processing your source</h2>
        </div>
        <button
          className="vn-label text-vn-dim transition-colors hover:text-vn-mist"
          onClick={onCancel}
          type="button"
        >
          Cancel
        </button>
      </div>

      <div className="divide-y divide-vn-ash border border-vn-ash">
        {rows.map((row) => (
          <div
            key={row.key}
            className={`flex items-center justify-between px-5 py-5 border-l-2 ${
              row.state === "active"
                ? "border-l-vn-amber"
                : row.state === "done"
                ? "border-l-vn-ash"
                : "border-l-transparent"
            } ${rowClass(row.state)}`}
          >
            <div className="flex items-center gap-4">
              {indicator(row.state)}
              <div>
                <span className="text-sm font-medium">{row.label}</span>
                {row.key === "download" && showDownloadHint ? (
                  <p className="mt-1 text-xs text-vn-dim leading-relaxed">
                    Loading your footage. The narrator works from real frames — this is what makes it accurate.
                  </p>
                ) : null}
              </div>
            </div>
            {row.detail ? (
              <span className="font-mono text-xs text-vn-mist">{row.detail}</span>
            ) : null}
          </div>
        ))}
      </div>
    </section>
  );
}
