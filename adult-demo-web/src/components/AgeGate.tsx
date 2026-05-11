"use client";

import { useState } from "react";

export default function AgeGate({ onConfirm }: { onConfirm: () => void }) {
  const [ageConfirmed, setAgeConfirmed] = useState(false);
  const [platformConfirmed, setPlatformConfirmed] = useState(false);
  const ready = ageConfirmed && platformConfirmed;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90">
      <div className="mx-4 w-full max-w-sm space-y-6 rounded-xl border border-zinc-700 bg-zinc-900 p-8">
        <h1 className="text-xl font-semibold text-white">Before you continue</h1>
        <label className="flex cursor-pointer items-start gap-3">
          <input
            type="checkbox"
            checked={ageConfirmed}
            onChange={(event) => setAgeConfirmed(event.target.checked)}
            className="mt-1 accent-white"
          />
          <span className="text-sm text-zinc-300">I am 18 years of age or older.</span>
        </label>
        <label className="flex cursor-pointer items-start gap-3">
          <input
            type="checkbox"
            checked={platformConfirmed}
            onChange={(event) => setPlatformConfirmed(event.target.checked)}
            className="mt-1 accent-white"
          />
          <span className="text-sm text-zinc-300">
            I am a platform operator, content producer, or accessibility professional accessing this demo in a
            professional capacity.
          </span>
        </label>
        <button
          onClick={onConfirm}
          disabled={!ready}
          className="w-full rounded-lg py-3 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:bg-zinc-700 disabled:text-zinc-500 enabled:bg-white enabled:text-black enabled:hover:bg-zinc-200"
          type="button"
        >
          Enter
        </button>
      </div>
    </div>
  );
}
