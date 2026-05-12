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
      className="flex flex-col gap-4 md:flex-row"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit(value);
      }}
    >
      <input
        className="min-h-14 flex-1 border-b-2 border-vn-ash bg-transparent px-1 pb-3 text-base text-vn-fog outline-none transition-colors placeholder:text-vn-dim focus:border-vn-amber"
        placeholder="YouTube, Vimeo, or direct video URL"
        type="url"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      <button
        className="min-h-14 bg-vn-amber px-8 font-body text-sm font-semibold uppercase tracking-[0.18em] text-vn-black transition-colors hover:bg-amber-400 disabled:cursor-not-allowed disabled:bg-vn-ash disabled:text-vn-dim"
        disabled={processing || !value.trim()}
        type="submit"
      >
        Generate Audio Description
      </button>
    </form>
  );
}
