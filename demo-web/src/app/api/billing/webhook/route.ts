import { NextResponse } from "next/server";
import type Stripe from "stripe";

import {
  getStripeCustomerId,
  getSubscriptionItemId,
  grantCheckoutEntitlement,
  isCheckoutPlan,
  resolvePlanFromPriceId,
  revokeSubscriptionEntitlement,
} from "@/lib/billing";
import { getStripe } from "@/lib/stripe";

export const runtime = "nodejs";

async function syncSubscriptionState(
  subscription: Stripe.Subscription,
  fallbackPlan?: unknown,
) {
  const userId = typeof subscription.metadata.userId === "string" ? subscription.metadata.userId : undefined;
  if (!userId) {
    return;
  }

  const planCandidate = typeof subscription.metadata.plan === "string" ? subscription.metadata.plan : fallbackPlan;
  const plan = isCheckoutPlan(planCandidate) ? planCandidate : resolvePlanFromPriceId(subscription.items.data[0]?.price?.id);
  if (!plan) {
    return;
  }

  const configuredPriceId =
    typeof subscription.metadata.priceId === "string"
      ? subscription.metadata.priceId
      : subscription.items.data[0]?.price?.id;

  await grantCheckoutEntitlement({
    customerId: getStripeCustomerId(subscription.customer),
    plan,
    subscriptionId: subscription.id,
    subscriptionItemId: getSubscriptionItemId(subscription.items.data, configuredPriceId),
    userId,
  });
}

export async function POST(request: Request) {
  const stripe = getStripe();
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  const signature = request.headers.get("stripe-signature");

  if (!webhookSecret || !signature) {
    return NextResponse.json({ error: "Stripe webhook is not configured." }, { status: 400 });
  }

  const payload = await request.text();

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(payload, signature, webhookSecret);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Webhook verification failed.";
    return NextResponse.json({ error: message }, { status: 400 });
  }

  switch (event.type) {
    case "checkout.session.completed": {
      const session = event.data.object as Stripe.Checkout.Session;
      const userId =
        (typeof session.metadata?.userId === "string" ? session.metadata.userId : undefined) ??
        session.client_reference_id ??
        undefined;
      const priceId = typeof session.metadata?.priceId === "string" ? session.metadata.priceId : undefined;
      const plan =
        (typeof session.metadata?.plan === "string" && isCheckoutPlan(session.metadata.plan)
          ? session.metadata.plan
          : resolvePlanFromPriceId(priceId)) ?? null;

      if (userId && plan) {
        let subscriptionId: string | undefined;
        let subscriptionItemId: string | undefined;

        if (session.mode === "subscription" && session.subscription) {
          const stripeSubscriptionId =
            typeof session.subscription === "string" ? session.subscription : session.subscription.id;
          const subscription = await stripe.subscriptions.retrieve(stripeSubscriptionId);
          subscriptionId = subscription.id;
          subscriptionItemId = getSubscriptionItemId(subscription.items.data, priceId);
        }

        await grantCheckoutEntitlement({
          customerId: getStripeCustomerId(session.customer),
          plan,
          sessionId: session.id,
          subscriptionId,
          subscriptionItemId,
          userId,
        });
      }
      break;
    }
    case "invoice.payment_succeeded": {
      const invoice = event.data.object as Stripe.Invoice;
      const invoiceWithSubscription = invoice as Stripe.Invoice & {
        subscription?: string | Stripe.Subscription | null;
      };
      const subscriptionId =
        typeof invoiceWithSubscription.subscription === "string"
          ? invoiceWithSubscription.subscription
          : invoiceWithSubscription.subscription?.id;

      if (subscriptionId) {
        const subscription = await stripe.subscriptions.retrieve(subscriptionId);
        await syncSubscriptionState(subscription);
      }
      break;
    }
    case "customer.subscription.deleted": {
      const subscription = event.data.object as Stripe.Subscription;
      const userId = typeof subscription.metadata.userId === "string" ? subscription.metadata.userId : undefined;
      if (userId) {
        await revokeSubscriptionEntitlement(userId);
      }
      break;
    }
    default:
      break;
  }

  return NextResponse.json({ received: true });
}
