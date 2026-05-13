import { auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

import { canCreateJob } from "@/lib/billing";

export const runtime = "nodejs";

function normalizeUsageMinutes(value: unknown) {
  const parsed = Number(value);

  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 90;
  }

  return Math.min(24 * 60, Math.ceil(parsed));
}

export async function POST(request: Request) {
  const { userId } = await auth();

  if (!userId) {
    return NextResponse.json({ error: "Authentication required." }, { status: 401 });
  }

  const entitlement = await canCreateJob(userId);
  if (!entitlement.allowed) {
    return NextResponse.json(
      {
        error: "An active billing entitlement is required before creating a job.",
        redirectTo: "/pricing",
      },
      { status: 402 },
    );
  }

  const payload = (await request.json().catch(() => ({}))) as {
    estimatedMinutes?: number;
    s3Key?: string;
    source?: string;
    sourceType?: string;
  };
  const jobId = crypto.randomUUID();
  const usageMinutes = normalizeUsageMinutes(payload.estimatedMinutes);

  const qs = new URLSearchParams({ minutes: String(usageMinutes) });
  if (typeof payload.s3Key === "string" && payload.s3Key) qs.set("s3Key", payload.s3Key);
  if (typeof payload.source === "string" && payload.source) qs.set("source", payload.source);

  return NextResponse.json({
    jobId,
    processingUrl: `/processing/${jobId}?${qs}`,
  });
}
