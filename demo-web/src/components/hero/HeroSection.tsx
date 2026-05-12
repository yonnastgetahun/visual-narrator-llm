"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { FilmFrameCanvas } from "./FilmFrameCanvas";

const EASE_VN = [0.16, 1, 0.3, 1] as const;

// Shared scroll-entrance variant — stagger handled by delay at call site
const scrollEntrance = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0 },
};

export function HeroSection() {
  return (
    <section className="relative min-h-[100dvh] overflow-hidden bg-vn-black">
      {/* Letterbox bars — 64px, inner edge amber gradient fade */}
      <div
        aria-hidden="true"
        className="absolute left-0 right-0 top-0 z-10 h-16"
        style={{
          background:
            "linear-gradient(to bottom, #0a0a0a 55%, rgba(10,10,10,0.0) 100%), linear-gradient(to bottom, #0a0a0a 0%, rgba(245,158,11,0.04) 100%)",
        }}
      />
      <div
        aria-hidden="true"
        className="absolute bottom-0 left-0 right-0 z-10 h-16"
        style={{
          background:
            "linear-gradient(to top, #0a0a0a 55%, rgba(10,10,10,0.0) 100%), linear-gradient(to top, #0a0a0a 0%, rgba(245,158,11,0.04) 100%)",
        }}
      />

      {/* Subtle vignette */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-[2]"
        style={{
          background:
            "radial-gradient(ellipse 90% 90% at 50% 50%, transparent 35%, rgba(10,10,10,0.75) 100%)",
        }}
      />

      {/* Three.js canvas — right half, deeper mask so left reads clean */}
      <div
        aria-hidden="true"
        className="absolute inset-y-0 right-0 z-[1] w-full md:w-[60%]"
        style={{
          maskImage:
            "linear-gradient(to right, transparent 0%, rgba(0,0,0,0.15) 18%, black 38%)",
          WebkitMaskImage:
            "linear-gradient(to right, transparent 0%, rgba(0,0,0,0.15) 18%, black 38%)",
        }}
      >
        <FilmFrameCanvas />
      </div>

      {/* Content — left column */}
      <div className="relative z-[5] flex min-h-[100dvh] flex-col justify-center px-6 pb-16 pt-16 md:pl-[8vw] md:pr-0">
        <div className="max-w-[640px]">
          {/* Eyebrow */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: EASE_VN, delay: 0.1 }}
            className="flex items-center gap-3"
          >
            <span className="vn-amber-rule" />
            <span className="vn-label text-vn-amber">Visual Narrator</span>
          </motion.div>

          {/* Headline — pushed to display limit */}
          <motion.h1
            initial={{ opacity: 0, y: 22 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, ease: EASE_VN, delay: 0.2 }}
            className="vn-display mt-5 text-vn-cream"
            style={{ letterSpacing: "-0.025em", lineHeight: "0.96" }}
          >
            Audio description.
            <br />
            <span className="text-vn-amber">In minutes,</span>
            <br />
            not weeks.
          </motion.h1>

          {/* Sub — scroll-triggered entrance */}
          <motion.p
            variants={scrollEntrance}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.7, ease: EASE_VN, delay: 0.1 }}
            className="mt-7 max-w-[46ch] text-[1.0625rem] leading-[1.72] text-vn-mist"
          >
            Upload a film or paste a URL. Get a broadcast-ready AD track —
            MP3, SRT, and compliance report — in about 20 minutes.{" "}
            <span className="text-vn-fog">Starting at $39.</span>
          </motion.p>

          {/* CTA — scroll-triggered */}
          <motion.div
            variants={scrollEntrance}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.65, ease: EASE_VN, delay: 0.2 }}
            className="mt-9 flex flex-wrap items-center gap-x-8 gap-y-4"
          >
            {/* Primary — solid amber button */}
            <Link href="/upload">
              <motion.span
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="inline-flex cursor-pointer items-center gap-2.5 bg-vn-amber px-7 py-3.5 text-[0.9375rem] font-medium text-vn-black transition-colors duration-200 hover:bg-amber-400"
                style={{ letterSpacing: "0.01em" }}
              >
                Describe a film
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                  <path
                    d="M2 7h10M8 3l4 4-4 4"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </motion.span>
            </Link>

            {/* Editorial link — amber underline, no button feel */}
            <Link
              href="#how-it-works"
              className="group relative text-[0.9375rem] text-vn-amber transition-colors duration-200 hover:text-amber-300"
              style={{ letterSpacing: "0.005em" }}
            >
              <span>Try it live</span>
              <span
                aria-hidden="true"
                className="absolute inset-x-0 -bottom-[2px] h-px bg-vn-amber opacity-60 transition-opacity duration-200 group-hover:opacity-100"
              />
              {" "}
              <span className="text-vn-fog transition-colors duration-200 group-hover:text-amber-300/70">&#8594;</span>
            </Link>
          </motion.div>

          {/* Stat line — scroll-triggered */}
          <motion.p
            variants={scrollEntrance}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.6, ease: EASE_VN, delay: 0.32 }}
            className="vn-label mt-5 text-vn-dim"
          >
            1,350+ films described&nbsp;&nbsp;·&nbsp;&nbsp;$39 flat rate
          </motion.p>
        </div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 0.8 }}
        className="absolute bottom-[4.5rem] left-[8vw] z-[5] hidden items-center gap-2.5 md:flex"
      >
        <motion.div
          animate={{ y: [0, 5, 0] }}
          transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
          className="h-8 w-[1px] bg-gradient-to-b from-vn-amber to-transparent"
        />
        <span className="vn-label text-vn-dim">Scroll</span>
      </motion.div>
    </section>
  );
}
