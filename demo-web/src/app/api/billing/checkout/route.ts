import { auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

import { getBaseUrl, getPriceIdForPlan, isRecurringPlan, resolvePlanFromPriceId } from "@/lib/billing";
import { getStripe } from "@/lib/stripe";

export const runtime = "nodejs";

function isJsonRequest(request: Request) {
  return request.headers.get("content-type")?.includes("application/json") ?? false;
}

async function readPriceId(request: Request) {
  if (isJsonRequest(request)) {
    const payload = (await request.json()) as { priceId?: string };
    return payload.priceId;
  }

  const formData = await request.formData();
  const value = formData.get("priceId");
  return typeof value === "string" ? value : undefined;
}

export async function POST(request: Request) {
  const isJson = isJsonRequest(request);
  const { userId } = await auth();

  if (!userId) {
    if (isJson) {
      return NextResponse.json({ error: "Authentication required." }, { status: 401 });
    }

    return NextResponse.redirect(new URL("/sign-in?redirect_url=/pricing", request.url), 303);
  }

  const priceId = await readPriceId(request);
  const plan = resolvePlanFromPriceId(priceId);

  if (!plan || !priceId || getPriceIdForPlan(plan) !== priceId) {
    return NextResponse.json({ error: "Unknown Stripe price." }, { status: 400 });
  }

  const stripe = getStripe();
  const baseUrl = getBaseUrl(new URL(request.url).origin);
  const session = await stripe.checkout.sessions.create({
    cancel_url: new URL("/pricing", baseUrl).toString(),
    client_reference_id: userId,
    customer_creation: plan === "per_film" ? "always" : undefined,
    line_items: [{ price: priceId, quantity: 1 }],
    metadata: {
      plan,
      priceId,
      userId,
    },
    mode: isRecurringPlan(plan) ? "subscription" : "payment",
    payment_method_types: ["card"],
    subscription_data: isRecurringPlan(plan)
      ? {
          metadata: {
            plan,
            priceId,
            userId,
          },
        }
      : undefined,
    success_url: new URL("/dashboard?checkout=success", baseUrl).toString(),
  });

  if (isJson) {
    return NextResponse.json({ url: session.url });
  }

  if (!session.url) {
    return NextResponse.json({ error: "Stripe session missing redirect URL." }, { status: 500 });
  }

  return NextResponse.redirect(session.url, 303);
}
