import { NextRequest } from "next/server";

const API_BASE =
  process.env.ADULT_DEMO_API_BASE_URL ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const ADULT_DEMO_SHARED_KEY = process.env.ADULT_DEMO_SHARED_KEY;

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function sseHeaders() {
  return {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
  };
}

function errorStream(message: string, status = 500) {
  return new Response(`event: error\ndata: ${JSON.stringify({ message })}\n\n`, {
    status,
    headers: sseHeaders(),
  });
}

export async function GET(request: NextRequest) {
  if (!ADULT_DEMO_SHARED_KEY) {
    return errorStream("Adult demo proxy is not configured.", 500);
  }

  const upstreamUrl = new URL("/api/ad-adult", API_BASE);
  request.nextUrl.searchParams.forEach((value, key) => {
    upstreamUrl.searchParams.set(key, value);
  });

  const upstreamResponse = await fetch(upstreamUrl, {
    headers: {
      Accept: "text/event-stream",
      "x-demo-key": ADULT_DEMO_SHARED_KEY,
    },
    cache: "no-store",
  });

  if (!upstreamResponse.ok || !upstreamResponse.body) {
    let message = `Adult demo upstream returned status ${upstreamResponse.status}.`;

    try {
      const contentType = upstreamResponse.headers.get("content-type") ?? "";
      if (contentType.includes("application/json")) {
        const payload = (await upstreamResponse.json()) as { detail?: string };
        if (payload.detail) {
          message = payload.detail;
        }
      } else {
        const text = (await upstreamResponse.text()).trim();
        if (text) {
          message = text;
        }
      }
    } catch {
      // Keep the fallback message above.
    }

    return errorStream(message, upstreamResponse.status);
  }

  return new Response(upstreamResponse.body, {
    status: 200,
    headers: sseHeaders(),
  });
}
