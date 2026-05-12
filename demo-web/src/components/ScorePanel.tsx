"use client";

import { formatClock } from "@/lib/format";
import type { ScoreReport } from "@/lib/types";

type ScorePanelProps = {
  onScore: () => void;
  scoring: boolean;
  scoringProgress: { current: number; total: number } | null;
  report: ScoreReport | null;
};

function ScoreLine({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between border-b border-vn-ash py-4 text-sm">
      <span className="text-vn-mist">{label}</span>
      <span className="font-mono text-vn-fog">{value.toFixed(1)} / 10</span>
    </div>
  );
}

export function ScorePanel({ onScore, scoring, scoringProgress, report }: ScorePanelProps) {
  if (!report) {
    return (
      <section className="bg-vn-ink border border-vn-ash p-8">
        <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div>
            <span className="vn-label text-vn-amber flex items-center gap-2.5 mb-4">
              <span className="vn-amber-rule" />
              Quality Review
            </span>
            <h2 className="font-display text-2xl text-vn-cream leading-tight">AD Quality Score</h2>
            <p className="mt-2 text-sm text-vn-mist leading-relaxed">
              Run the quality reviewer against every narration after generation completes.
            </p>
          </div>
          <button
            className="self-start md:self-center bg-vn-amber px-8 py-4 font-body text-sm font-semibold uppercase tracking-[0.18em] text-vn-black transition-colors hover:bg-amber-400 disabled:cursor-not-allowed disabled:bg-vn-ash disabled:text-vn-dim"
            disabled={scoring}
            onClick={onScore}
            type="button"
          >
            {scoring && scoringProgress
              ? `Scoring ${scoringProgress.current} / ${scoringProgress.total}`
              : "Score This AD Track"}
          </button>
        </div>
      </section>
    );
  }

  const flagged = report.scores.filter((score) => score.flag);

  return (
    <section className="bg-vn-ink border border-vn-ash p-8">
      <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between mb-8">
        <div>
          <span className="vn-label text-vn-amber flex items-center gap-2.5 mb-4">
            <span className="vn-amber-rule" />
            Scoring Complete
          </span>
          <h2 className="font-display text-2xl text-vn-cream leading-tight">AD Quality Score</h2>
        </div>
        <div className="border border-vn-amber/30 bg-vn-amber/10 px-6 py-4 text-right">
          <p className="vn-label text-vn-mist mb-1">Grade</p>
          <p className="font-display text-4xl text-vn-cream">{report.grade}</p>
          <p className="font-mono text-sm text-vn-fog mt-1">Overall {report.aggregate.overall.toFixed(1)} / 10</p>
        </div>
      </div>

      <div className="border-t border-vn-ash">
        <ScoreLine label="Accuracy" value={report.aggregate.accuracy} />
        <ScoreLine label="Relevance" value={report.aggregate.relevance} />
        <ScoreLine label="WCAG Compliance" value={report.aggregate.wcag_compliance} />
        <ScoreLine label="Conciseness" value={report.aggregate.conciseness} />
        <div className="flex items-center justify-between border-b border-vn-ash py-4 text-sm">
          <span className="text-vn-mist">Within word limit</span>
          <span className="font-mono text-vn-fog">{report.aggregate.within_limit_pct.toFixed(0)}%</span>
        </div>
        <div className="flex items-center justify-between border-b border-vn-ash py-4 text-sm">
          <span className="text-vn-mist">Present tense</span>
          <span className="font-mono text-vn-fog">{report.aggregate.tense_ok_pct.toFixed(0)}%</span>
        </div>
        <div className="flex items-center justify-between py-4 text-sm">
          <span className="text-vn-mist">Flagged</span>
          <span className="font-mono text-vn-fog">
            {report.flagged} of {report.scored}
          </span>
        </div>
      </div>

      <div className="mt-8">
        <span className="vn-label text-vn-mist flex items-center gap-2.5 mb-5">
          <span className="vn-amber-rule" />
          {flagged.length ? `Flagged Descriptions (${flagged.length})` : "Flagged Descriptions"}
        </span>
        {flagged.length ? (
          <div className="space-y-5">
            {flagged.map((score) => (
              <article
                key={`${score.srt_index}-${score.start_sec}`}
                className="border-l-2 border-vn-amber pl-5 py-1"
              >
                <p className="font-mono text-xs text-vn-amber tracking-[0.18em] uppercase">
                  {formatClock(score.start_sec)} → {formatClock(score.end_sec)} · overall={score.overall.toFixed(1)} · flagged
                </p>
                <p className="mt-2 text-sm text-vn-fog leading-relaxed">{score.description}</p>
                {score.flag_reason ? (
                  <p className="mt-2 text-xs text-vn-mist">{score.flag_reason}</p>
                ) : null}
              </article>
            ))}
          </div>
        ) : (
          <p className="text-sm text-vn-mist">All descriptions cleared the configured threshold.</p>
        )}
      </div>
    </section>
  );
}
