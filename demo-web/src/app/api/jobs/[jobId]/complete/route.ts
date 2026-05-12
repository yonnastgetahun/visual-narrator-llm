import { auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

import { finalizeCompletedJob } from "@/lib/billing";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const { userId } = await auth();

  if (!userId) {
    return NextResponse.json({ error: "Authentication required." }, { status: 401 });
  }

  const payload = (await request.json().catch(() => ({}))) as { estimatedMinutes?: number };
  const result = await finalizeCompletedJob(userId, payload.estimatedMinutes);

  return NextResponse.json({
    ok: true,
    ...result,
  });
}
