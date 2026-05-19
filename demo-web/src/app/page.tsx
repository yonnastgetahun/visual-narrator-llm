"use client";

import { SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import { motion } from "framer-motion";
import Link from "next/link";
import { HeroSection } from "@/components/hero/HeroSection";

const EASE_VN = [0.16, 1, 0.3, 1] as const;
const CLERK_ENABLED = Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY);

const HOW_IT_WORKS = [
  {
    step: "01",
    title: "Upload or paste",
    body: "Drop a video file or paste a YouTube, Vimeo, or direct URL. We accept up to 4 GB.",
  },
  {
    step: "02",
    title: "We detect the gaps",
    body: "Our pipeline finds every moment where dialogue pauses — the exact windows for audio description.",
  },
  {
    step: "03",
    title: "GPT-4o describes each frame",
    body: "Real frames are sampled from each gap. The model writes tight, present-tense AD copy per WCAG 2.1.",
  },
  {
    step: "04",
    title: "Audio is synthesised and mixed",
    body: "Each narration is rendered to voice, timed precisely, and mixed into a broadcast-ready MP3.",
  },
];

const AUDIENCES = [
  { title: "Indie filmmakers",       body: "Professional AD for your festival cut — without a $3,000 studio booking." },
  { title: "EdTech platforms",       body: "WCAG 2.1 AA compliance for your course library, at scale." },
  { title: "Corporate video teams",  body: "Turn around AD for training and marketing content same day." },
  { title: "AD studios",             body: "Cut production time from weeks to hours. Handle 10x the volume." },
];

export default function LandingPage() {
  return (
    <main>
      <header className="fixed inset-x-0 top-0 z-30 border-b border-vn-line bg-vn-black/95 backdrop-blur">
        <div className="flex items-center justify-between px-6 py-4 md:px-[8vw]">
          <Link href="/" className="vn-label text-vn-amber transition-colors hover:text-amber-300">
            Visual Narrator
          </Link>
          <div className="flex items-center">
            {CLERK_ENABLED ? (
              <>
                <SignedIn>
                  <UserButton
                    afterSignOutUrl="/"
                    appearance={{
                      elements: {
                        userButtonAvatarBox: "h-9 w-9",
                      },
                    }}
                  />
                </SignedIn>
                <SignedOut>
                  <Link
                    href="/sign-in"
                    className="vn-label text-vn-mist transition-colors hover:text-vn-cream"
                  >
                    Sign in
                  </Link>
                </SignedOut>
              </>
            ) : (
              <Link
                href="/demo"
                className="vn-label text-vn-mist transition-colors hover:text-vn-cream"
              >
                Live demo
              </Link>
            )}
          </div>
        </div>
      </header>

      <HeroSection />

      {/* How it works */}
      <section id="how-it-works" className="bg-vn-ink px-6 py-28 md:px-[8vw]">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.7, ease: EASE_VN }}
          className="mb-14"
        >
          <span className="vn-label text-vn-amber flex items-center gap-2.5 mb-3">
            <span className="vn-amber-rule" />
            How it works
          </span>
          <h2 className="vn-title text-vn-cream max-w-[18ch]">Four steps. One AD track.</h2>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-16 gap-y-12 max-w-[960px]">
          {HOW_IT_WORKS.map((item, i) => (
            <motion.div
              key={item.step}
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.6, ease: EASE_VN, delay: i * 0.09 }}
              className="flex gap-6"
            >
              <span className="font-mono text-[2.25rem] text-vn-amber leading-none font-medium flex-none">{item.step}</span>
              <div>
                <h3 className="text-[1.0625rem] font-medium text-vn-cream mb-2">{item.title}</h3>
                <p className="text-[0.9375rem] leading-relaxed text-vn-mist">{item.body}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Audiences */}
      <section className="bg-vn-black px-6 py-28 md:px-[8vw]">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.7, ease: EASE_VN }}
          className="mb-14"
        >
          <span className="vn-label text-vn-amber flex items-center gap-2.5 mb-3">
            <span className="vn-amber-rule" />
            Built for
          </span>
          <h2 className="vn-title text-vn-cream max-w-[22ch]">People who make films and need them described.</h2>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-px bg-vn-line max-w-[960px]">
          {AUDIENCES.map((a, i) => (
            <motion.div
              key={a.title}
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.5, delay: i * 0.07 }}
              className="bg-vn-black px-7 py-8"
            >
              <h3 className="text-[1.0625rem] font-medium text-vn-cream mb-2">{a.title}</h3>
              <p className="text-[0.9375rem] leading-relaxed text-vn-mist">{a.body}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Pricing CTA */}
      <section className="bg-vn-ink px-6 py-28 md:px-[8vw] border-t border-vn-line">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.7, ease: EASE_VN }}
          className="max-w-[640px]"
        >
          <span className="vn-label text-vn-amber flex items-center gap-2.5 mb-6">
            <span className="vn-amber-rule" />
            Pricing
          </span>
          <div className="flex items-end gap-3 mb-2">
            <span className="font-mono text-[4rem] text-vn-cream font-medium leading-none">$39</span>
            <span className="text-vn-mist mb-2">per 90-minute film</span>
          </div>
          <p className="text-[0.9375rem] leading-relaxed text-vn-mist mb-8">
            Flat rate. No subscription. Includes MP3 audio track, SRT subtitles, and a full WCAG 2.1 compliance report. Charged only on successful completion.
          </p>
          <div className="flex flex-col sm:flex-row items-start gap-6">
            <Link href="/demo">
              <motion.span
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.98 }}
                className="inline-flex cursor-pointer items-center gap-2.5 bg-vn-amber px-8 py-4 text-[1rem] font-medium text-vn-black hover:bg-amber-400 transition-colors duration-200"
              >
                Describe a film
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                  <path d="M2 7h10M8 3l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </motion.span>
            </Link>
            <Link href="/demo" className="self-center text-[0.9375rem] text-vn-mist hover:text-vn-fog transition-colors duration-200">
              Watch it run live →
            </Link>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="bg-vn-black border-t border-vn-line px-6 py-8 md:px-[8vw] flex flex-col md:flex-row items-center justify-between gap-4">
        <span className="vn-label text-vn-dim">Visual Narrator — Audio Description Platform</span>
        <span className="vn-label text-vn-dim">app.vnpoverview.com</span>
      </footer>
    </main>
  );
}
