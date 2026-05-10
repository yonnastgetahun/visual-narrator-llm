"use client";

import { formatClock } from "@/lib/format";
import type { ScoreReport } from "@/lib/types";

type ScorePanelProps = {
  onScore: () => void;
  scoring: boolean;
  scoringProgress: { current: number; total: number } | null;
  report: ScoreReport | null;
};

function scoreLine(label: string, value: number) {
  return (
    <div className="flex items-center justify-between border-b border-white/10 py-3 text-sm text-slate-200">
      <span>{label}</span>
      <span className="font-mono">{value.toFixed(1)} / 10</span>
    </div>
  );
}

export function ScorePanel({ onScore, scoring, scoringProgress, report }: ScorePanelProps) {
  if (!report) {
    return (
      <section className="rounded-[2rem] border border-white/10 bg-slate-950/65 p-6 shadow-2xl shadow-cyan-950/20 backdrop-blur">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.32em] text-cyan-300">Scoring</p>
            <h2 className="mt-2 text-2xl font-semibold text-white">AD Quality Score</h2>
            <p className="mt-2 text-sm text-slate-300">Run the quality reviewer against every narration after generation completes.</p>
          </div>
          <button
            className="rounded-2xl bg-cyan-400 px-6 py-4 text-sm font-semibold uppercase tracking-[0.18em] text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
            disabled={scoring}
            onClick={onScore}
            type="button"
          >
            {scoring && scoringProgress ? `Scoring ${scoringProgress.current} / ${scoringProgress.total}` : "Score This AD Track"}
          </button>
        </div>
      </section>
    );
  }

  const flagged = report.scores.filter((score) => score.flag);

  return (
    <section className="rounded-[2rem] border border-white/10 bg-slate-950/65 p-6 shadow-2xl shadow-cyan-950/20 backdrop-blur">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.32em] text-cyan-300">Scoring Complete</p>
          <h2 className="mt-2 text-2xl font-semibold text-white">AD Quality Score</h2>
        </div>
        <div className="rounded-2xl border border-emerald-300/30 bg-emerald-300/10 px-5 py-4 text-right">
          <p className="text-xs uppercase tracking-[0.26em] text-emerald-200">Grade</p>
          <p className="mt-1 text-3xl font-semibold text-white">{report.grade}</p>
          <p className="font-mono text-sm text-emerald-100">Overall {report.aggregate.overall.toFixed(1)} / 10</p>
        </div>
      </div>

      <div className="mt-6">
        {scoreLine("Accuracy", report.aggregate.accuracy)}
        {scoreLine("Relevance", report.aggregate.relevance)}
        {scoreLine("WCAG Compliance", report.aggregate.wcag_compliance)}
        {scoreLine("Conciseness", report.aggregate.conciseness)}
        <div className="flex items-center justify-between py-3 text-sm text-slate-200">
          <span>Within word limit</span>
          <span className="font-mono">{report.aggregate.within_limit_pct.toFixed(0)}%</span>
        </div>
        <div className="flex items-center justify-between py-3 text-sm text-slate-200">
          <span>Present tense</span>
          <span className="font-mono">{report.aggregate.tense_ok_pct.toFixed(0)}%</span>
        </div>
        <div className="flex items-center justify-between py-3 text-sm text-slate-200">
          <span>Flagged</span>
          <span className="font-mono">
            {report.flagged} of {report.scored}
          </span>
        </div>
      </div>

      <div className="mt-6">
        <h3 className="text-lg font-medium text-white">
          {flagged.length ? `Flagged Descriptions (${flagged.length})` : "No flagged descriptions"}
        </h3>
        {flagged.length ? (
          <div className="mt-4 space-y-4">
            {flagged.map((score) => (
              <article key={`${score.srt_index}-${score.start_sec}`} className="rounded-2xl border border-amber-300/30 bg-amber-300/10 p-4">
                <p className="font-mono text-xs uppercase tracking-[0.22em] text-amber-100">
                  {formatClock(score.start_sec)} → {formatClock(score.end_sec)} · overall={score.overall.toFixed(1)} · ✗
                </p>
                <p className="mt-3 text-slate-100">{score.description}</p>
                {score.flag_reason ? <p className="mt-3 text-sm text-amber-100">↳ {score.flag_reason}</p> : null}
              </article>
            ))}
          </div>
        ) : (
          <p className="mt-3 text-sm text-slate-300">All descriptions cleared the configured threshold.</p>
        )}
      </div>
    </section>
  );
}
