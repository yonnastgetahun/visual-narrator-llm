/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // --- VN Design Tokens ---
      colors: {
        vn: {
          black:   "var(--vn-black)",
          ink:     "var(--vn-ink)",
          carbon:  "var(--vn-carbon)",
          ash:     "var(--vn-ash)",
          dim:     "var(--vn-dim)",
          mist:    "var(--vn-mist)",
          fog:     "var(--vn-fog)",
          cream:   "var(--vn-cream)",
          amber:   "#F59E0B",
          "amber-dim": "#C27D08",
          "amber-glow": "rgba(245,158,11,0.15)",
          "amber-border": "rgba(245,158,11,0.25)",
          line:    "var(--vn-line)",
          "line-amber": "rgba(245,158,11,0.20)",
        },
      },
      fontFamily: {
        display: ["var(--font-playfair)", "Georgia", "serif"],
        body:    ["var(--font-geist)", "system-ui", "sans-serif"],
        mono:    ["var(--font-geist-mono)", "monospace"],
      },
      fontSize: {
        "vn-hero":  ["clamp(3rem, 6vw, 5.5rem)", { lineHeight: "1.02", letterSpacing: "-0.03em" }],
        "vn-title": ["clamp(2rem, 4vw, 3.5rem)", { lineHeight: "1.05", letterSpacing: "-0.025em" }],
        "vn-label": ["0.6875rem",                { lineHeight: "1",    letterSpacing: "0.22em" }],
      },
      spacing: {
        letterbox: "3.5rem",
      },
      transitionTimingFunction: {
        "vn-ease":   "cubic-bezier(0.16, 1, 0.3, 1)",
        "vn-in":     "cubic-bezier(0.4, 0, 1, 1)",
        "vn-out":    "cubic-bezier(0, 0, 0.2, 1)",
        "vn-spring": "cubic-bezier(0.34, 1.56, 0.64, 1)",
      },
      transitionDuration: {
        "400": "400ms",
        "600": "600ms",
        "800": "800ms",
      },
      animation: {
        "grain":     "grain 8s steps(10) infinite",
        "pulse-amber": "pulse-amber 3s ease-in-out infinite",
        "float":     "float 6s ease-in-out infinite",
        "waveform":  "waveform 1.4s ease-in-out infinite",
        "shimmer":   "shimmer 2.2s linear infinite",
      },
      keyframes: {
        grain: {
          "0%, 100%": { transform: "translate(0,0)" },
          "10%":  { transform: "translate(-2%,-2%)" },
          "20%":  { transform: "translate(2%,-1%)" },
          "30%":  { transform: "translate(-1%,2%)" },
          "40%":  { transform: "translate(2%,2%)" },
          "50%":  { transform: "translate(-2%,1%)" },
          "60%":  { transform: "translate(1%,-2%)" },
          "70%":  { transform: "translate(-1%,1%)" },
          "80%":  { transform: "translate(2%,-1%)" },
          "90%":  { transform: "translate(-2%,2%)" },
        },
        "pulse-amber": {
          "0%, 100%": { opacity: "0.7" },
          "50%":      { opacity: "1" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%":      { transform: "translateY(-12px)" },
        },
        waveform: {
          "0%, 100%": { transform: "scaleY(0.3)" },
          "50%":      { transform: "scaleY(1)" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      backgroundImage: {
        "shimmer-line": "linear-gradient(90deg, transparent 0%, rgba(245,158,11,0.25) 50%, transparent 100%)",
      },
      boxShadow: {
        "vn-amber":  "0 0 0 1px rgba(245,158,11,0.3), 0 8px 32px rgba(245,158,11,0.08)",
        "vn-card":   "0 1px 0 rgba(255,255,255,0.04) inset, 0 16px 48px rgba(0,0,0,0.5)",
        "vn-lift":   "0 24px 64px rgba(0,0,0,0.6), 0 1px 0 rgba(255,255,255,0.06) inset",
      },
    },
  },
  plugins: [],
};
