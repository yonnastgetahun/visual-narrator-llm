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

  return (
    <section className="bg-vn-ink border border-vn-ash p-8">
      {/* Flat underline tabs */}
      <div className="flex border-b border-vn-ash mb-8">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            className={`pb-3 pr-8 text-sm transition-colors ${
              activeTab === tab.key
                ? "border-b-2 border-vn-amber -mb-px text-vn-cream font-medium"
                : "text-vn-dim hover:text-vn-mist"
            }`}
            onClick={() => setActiveTab(tab.key)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div>
        {activeTab === "audio" ? <AudioPlayer narrations={manifest.narrations} /> : null}
        {activeTab === "srt" ? (
          <pre className="max-h-[32rem] overflow-auto bg-vn-carbon p-5 font-mono text-sm text-vn-fog leading-relaxed">
            {srt}
          </pre>
        ) : null}
        {activeTab === "manifest" ? (
          <pre className="max-h-[32rem] overflow-auto bg-vn-carbon p-5 font-mono text-sm text-vn-fog leading-relaxed">
            {manifestJson}
          </pre>
        ) : null}
      </div>

      <div className="mt-8 border-t border-vn-ash pt-5">
        <span className="font-mono text-xs text-vn-mist">
          GPT ${manifest.gpt_cost_estimate.toFixed(3)} · TTS ${manifest.tts_cost_estimate.toFixed(3)} · Total ${manifest.total_cost_estimate.toFixed(3)}
        </span>
      </div>
    </section>
  );
}
