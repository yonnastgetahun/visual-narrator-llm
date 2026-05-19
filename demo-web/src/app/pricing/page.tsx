import Link from "next/link";
import { auth } from "@clerk/nextjs/server";

import { PLAN_COPY, type CheckoutPlan, getPriceIdForPlan } from "@/lib/billing";

export const metadata = {
  title: "Pricing — Visual Narrator",
  description: "Choose a Visual Narrator plan for per-film, subscription, or metered API access.",
};

const TIERS: Array<{
  plan: CheckoutPlan;
  eyebrow: string;
  description: string;
  features: string[];
  recommended?: boolean;
}> = [
  {
    plan: "per_film",
    eyebrow: "For occasional releases",
    description: "One completed AD run for a single feature, short, or training film.",
    features: ["1 film credit per checkout", "MP3 audio track, SRT, and compliance report", "No recurring billing"],
  },
  {
    plan: "monthly",
    eyebrow: "For active teams",
    description: "Unlimited job creation for editorial, QA, and compliance teams working every week.",
    features: ["Unlimited web jobs", "Priority support", "Best fit for indie studios and internal teams"],
    recommended: true,
  },
  {
    plan: "annual",
    eyebrow: "For committed pipelines",
    description: "Lower blended cost for organizations that need AD on a stable yearly budget.",
    features: ["Unlimited web jobs", "Annual savings versus monthly", "Procurement-friendly fixed spend"],
  },
  {
    plan: "api_metered",
    eyebrow: "For product integrations",
    description: "Metered API access for platforms that need AD generated programmatically.",
    features: ["Billed by completed minutes", "Usage recorded after each finished job", "Designed for backend automation"],
  },
];

export default async function PricingPage() {
  const clerkEnabled = Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY && process.env.CLERK_SECRET_KEY);
  const { userId } = clerkEnabled ? await auth() : { userId: null };

  return (
    <main className="min-h-[100dvh] bg-vn-black px-6 py-20 md:px-[8vw]">
      <div className="mx-auto max-w-6xl">
        <div className="flex flex-wrap items-center justify-between gap-5 border-b border-vn-line pb-8">
          <div>
            <span className="vn-label flex items-center gap-2.5 text-vn-amber">
              <span className="vn-amber-rule" />
              Pricing
            </span>
            <h1 className="vn-title mt-5 max-w-[12ch] text-vn-cream">Choose how Visual Narrator bills your work.</h1>
            <p className="mt-4 max-w-[62ch] text-[0.975rem] leading-relaxed text-vn-mist">
              Same output pipeline, different buying motion. Use one-off film credits, a flat subscription,
              or metered API usage, all wired to Stripe prices from environment variables.
            </p>
          </div>
          <div className="text-right">
            <p className="vn-label text-vn-dim">{userId ? "Signed in" : "Sign in required at checkout"}</p>
            <Link href="/upload" className="mt-3 inline-flex text-[0.9375rem] text-vn-mist transition-colors hover:text-vn-cream">
              Back to upload
            </Link>
          </div>
        </div>

        <div className="mt-12 grid gap-5 lg:grid-cols-4">
          {TIERS.map((tier) => {
            const copy = PLAN_COPY[tier.plan];
            const priceId = getPriceIdForPlan(tier.plan);

            return (
              <section
                key={tier.plan}
                className={`flex min-h-[430px] flex-col border px-6 py-7 ${
                  tier.recommended
                    ? "border-vn-amber bg-[linear-gradient(180deg,rgba(245,158,11,0.16)_0%,rgba(20,20,20,0.92)_48%,rgba(10,10,10,1)_100%)]"
                    : "border-vn-line bg-vn-ink"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="vn-label text-vn-dim">{tier.eyebrow}</p>
                    <h2 className="mt-3 text-[1.35rem] font-medium text-vn-cream">{copy.label}</h2>
                  </div>
                  {tier.recommended ? (
                    <span className="vn-label border border-vn-amber-border bg-vn-amber-glow px-2.5 py-1 text-vn-amber">
                      Recommended
                    </span>
                  ) : null}
                </div>

                <div className="mt-8 flex items-end gap-2">
                  <span className="font-mono text-[2.65rem] leading-none text-vn-cream">{copy.priceLabel}</span>
                  <span className="pb-1 text-[0.95rem] text-vn-mist">
                    {tier.plan === "api_metered" ? "/ min" : tier.plan === "monthly" ? "/ mo" : tier.plan === "annual" ? "/ yr" : ""}
                  </span>
                </div>

                <p className="mt-4 text-[0.9375rem] leading-relaxed text-vn-mist">{tier.description}</p>

                <ul className="mt-8 flex flex-1 flex-col gap-3 text-[0.9rem] text-vn-fog">
                  {tier.features.map((feature) => (
                    <li key={feature} className="flex items-start gap-3">
                      <span className="mt-[0.35rem] h-1.5 w-1.5 rounded-full bg-vn-amber" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>

                <form action="/api/billing/checkout" method="POST" className="mt-8">
                  <input name="priceId" type="hidden" value={priceId ?? ""} />
                  <button
                    className="w-full bg-vn-amber px-4 py-3 text-[0.9375rem] font-medium text-vn-black transition-colors hover:bg-amber-400 disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={!priceId}
                    type="submit"
                  >
                    {userId ? `Checkout ${copy.label}` : "Sign in to checkout"}
                  </button>
                </form>

                <p className="mt-3 text-[0.8125rem] text-vn-dim">
                  {priceId
                    ? `Uses ${copy.intervalLabel} Stripe pricing.`
                    : "Missing Stripe price env var for this tier."}
                </p>
              </section>
            );
          })}
        </div>
      </div>
    </main>
  );
}
