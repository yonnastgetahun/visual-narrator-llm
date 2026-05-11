"use client";

import { useState } from "react";

import { AudioPlayer } from "@/components/AudioPlayer";
import { buildSrt, manifestWithoutAudio } from "@/lib/format";
import type { Manifest } from "@/lib/types";

type ResultsPanelProps = {
  manifest: Manifest;
};

type TabKey = "audio" | "srt" | "manifest";

const tabs: Array<{ key: TabKey; label: string }> = [
  { key: "audio", label: "Audio Player" },
  { key: "srt", label: "SRT Track" },
  { key: "manifest", label: "Manifest" },
];

export function ResultsPanel({ manifest }: ResultsPanelProps) {
  const [activeTab, setActiveTab] = useState<TabKey>("audio");
  const srt = buildSrt(manifest.narrations);
  const manifestJson = JSON.stringify(manifestWithoutAudio(manifest), null, 2);
  const visionCost = manifest.vision_cost_estimate ?? manifest.gpt_cost_estimate;

  return (
    <section className="rounded-[2rem] border border-white/10 bg-slate-950/65 p-6 shadow-2xl shadow-cyan-950/20 backdrop-blur">
      <div className="flex flex-wrap gap-2">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            className={`rounded-full px-4 py-2 text-sm transition ${
              activeTab === tab.key ? "bg-cyan-300 text-slate-950" : "bg-white/5 text-slate-300 hover:bg-white/10"
            }`}
            onClick={() => setActiveTab(tab.key)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="mt-6">
        {activeTab === "audio" ? <AudioPlayer narrations={manifest.narrations} /> : null}
        {activeTab === "srt" ? <pre className="max-h-[32rem] overflow-auto rounded-2xl bg-black/30 p-4 text-sm text-slate-200">{srt}</pre> : null}
        {activeTab === "manifest" ? (
          <pre className="max-h-[32rem] overflow-auto rounded-2xl bg-black/30 p-4 text-sm text-slate-200">{manifestJson}</pre>
        ) : null}
      </div>

      <div className="mt-6 rounded-2xl border border-cyan-300/20 bg-cyan-300/10 px-4 py-4 font-mono text-sm text-cyan-50">
        Vision: ${visionCost.toFixed(3)} · TTS: ${manifest.tts_cost_estimate.toFixed(3)} · Total: $
        {manifest.total_cost_estimate.toFixed(3)}
      </div>
    </section>
  );
}
