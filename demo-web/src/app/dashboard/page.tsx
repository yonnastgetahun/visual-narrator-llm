import Link from "next/link";
import { currentUser } from "@clerk/nextjs/server";

import { PLAN_COPY, getBillingState } from "@/lib/billing";

export const metadata = {
  title: "Dashboard — Visual Narrator",
};

export default async function DashboardPage({
  searchParams,
}: {
  searchParams?: { checkout?: string };
}) {
  const user = await currentUser();

  if (!user) {
    return null;
  }

  const state = getBillingState(user);
  const plan = state.publicMetadata.plan;
  const planLabel = plan === "free" ? "No active plan" : PLAN_COPY[plan].label;

  return (
    <main className="min-h-[100dvh] bg-vn-black px-6 py-20 md:px-[8vw]">
      <div className="mx-auto max-w-3xl">
        <span className="vn-label flex items-center gap-2.5 text-vn-amber">
          <span className="vn-amber-rule" />
          Dashboard
        </span>

        <h1 className="vn-title mt-5 text-vn-cream">Billing and access</h1>
        <p className="mt-3 max-w-[58ch] text-[0.975rem] leading-relaxed text-vn-mist">
          Review your current Visual Narrator entitlement before creating the next job.
        </p>

        {searchParams?.checkout === "success" ? (
          <div className="mt-8 border border-vn-amber-border bg-vn-amber-glow px-5 py-4 text-[0.9375rem] text-vn-fog">
            Checkout succeeded. Stripe will finish provisioning access through the webhook shortly.
          </div>
        ) : null}

        <section className="mt-8 grid gap-4 border border-vn-line bg-vn-ink p-6 md:grid-cols-3">
          <div>
            <p className="vn-label text-vn-dim">Plan</p>
            <p className="mt-2 text-[1.1rem] font-medium text-vn-cream">{planLabel}</p>
          </div>
          <div>
            <p className="vn-label text-vn-dim">Credits</p>
            <p className="mt-2 font-mono text-[1.1rem] text-vn-cream">{state.publicMetadata.credits}</p>
          </div>
          <div>
            <p className="vn-label text-vn-dim">Customer</p>
            <p className="mt-2 break-all font-mono text-[0.9rem] text-vn-mist">
              {state.privateMetadata.stripeCustomerId ?? "Pending webhook"}
            </p>
          </div>
        </section>

        <div className="mt-8 flex flex-wrap gap-4">
          <Link
            className="inline-flex items-center bg-vn-amber px-6 py-3 text-[0.9375rem] font-medium text-vn-black transition-colors hover:bg-amber-400"
            href="/upload"
          >
            Create a job
          </Link>
          <Link
            className="inline-flex items-center border border-vn-line px-6 py-3 text-[0.9375rem] text-vn-mist transition-colors hover:border-vn-amber hover:text-vn-cream"
            href="/pricing"
          >
            Change plan
          </Link>
        </div>
      </div>
    </main>
  );
}
