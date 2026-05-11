"use client";

type UrlInputProps = {
  value: string;
  processing: boolean;
  onChange: (value: string) => void;
  onSubmit: (url: string) => void;
};

export function UrlInput({ value, processing, onChange, onSubmit }: UrlInputProps) {
  return (
    <form
      className="flex flex-col gap-3 md:flex-row"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit(value);
      }}
    >
      <input
        className="min-h-14 flex-1 rounded-2xl border border-white/15 bg-slate-950/70 px-5 text-base text-white outline-none transition focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/30"
        placeholder="Direct video URL (first 3 minutes used)"
        type="url"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      <button
        className="min-h-14 rounded-2xl bg-cyan-400 px-6 text-sm font-semibold uppercase tracking-[0.18em] text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        disabled={processing || !value.trim()}
        type="submit"
      >
        Generate AD Track
      </button>
    </form>
  );
}
