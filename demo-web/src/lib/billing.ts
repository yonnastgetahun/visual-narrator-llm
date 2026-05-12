import { clerkClient } from "@clerk/nextjs/server";

import { getStripe } from "@/lib/stripe";

export type BillingPlan = "free" | "per_film" | "monthly" | "annual" | "api_metered";
export type CheckoutPlan = Exclude<BillingPlan, "free">;

export type BillingPublicMetadata = {
  plan: BillingPlan;
  credits: number;
};

export type BillingPrivateMetadata = {
  stripeCustomerId?: string;
  stripeSubscriptionId?: string;
  stripeSubscriptionItemId?: string;
  lastCheckoutSessionId?: string;
};

type BillingUserLike = {
  publicMetadata: unknown;
  privateMetadata: unknown;
};

const PRICE_ENV_NAMES: Record<CheckoutPlan, string[]> = {
  per_film: ["STRIPE_PRICE_PER_FILM"],
  monthly: ["STRIPE_PRICE_MONTHLY"],
  annual: ["STRIPE_PRICE_ANNUAL"],
  api_metered: ["STRIPE_PRICE_API_METERED", "STRIPE_PRICE_API"],
};

export const PLAN_COPY: Record<CheckoutPlan, { label: string; priceLabel: string; intervalLabel: string }> = {
  per_film: {
    label: "Per Film",
    priceLabel: "$39",
    intervalLabel: "per film",
  },
  monthly: {
    label: "Monthly",
    priceLabel: "$199",
    intervalLabel: "per month",
  },
  annual: {
    label: "Annual",
    priceLabel: "$1,499",
    intervalLabel: "per year",
  },
  api_metered: {
    label: "API",
    priceLabel: "$0.08",
    intervalLabel: "per minute",
  },
};

function asObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }

  return value as Record<string, unknown>;
}

function asString(value: unknown) {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function asPositiveInteger(value: unknown) {
  const parsed = Number(value);

  if (!Number.isFinite(parsed) || parsed < 0) {
    return 0;
  }

  return Math.floor(parsed);
}

export function isCheckoutPlan(value: unknown): value is CheckoutPlan {
  return value === "per_film" || value === "monthly" || value === "annual" || value === "api_metered";
}

export function isRecurringPlan(plan: CheckoutPlan) {
  return plan !== "per_film";
}

export function getPriceIdForPlan(plan: CheckoutPlan) {
  for (const envName of PRICE_ENV_NAMES[plan]) {
    const value = asString(process.env[envName]);
    if (value) {
      return value;
    }
  }

  return null;
}

export function resolvePlanFromPriceId(priceId: string | null | undefined): CheckoutPlan | null {
  if (!priceId) {
    return null;
  }

  const plans = Object.keys(PRICE_ENV_NAMES) as CheckoutPlan[];

  for (const plan of plans) {
    if (getPriceIdForPlan(plan) === priceId) {
      return plan;
    }
  }

  return null;
}

export function getBillingState(user: BillingUserLike) {
  const publicMetadata = asObject(user.publicMetadata);
  const privateMetadata = asObject(user.privateMetadata);
  const rawPlan = publicMetadata.plan;

  return {
    publicMetadata: {
      plan: rawPlan === "free" || isCheckoutPlan(rawPlan) ? rawPlan : "free",
      credits: asPositiveInteger(publicMetadata.credits),
    } satisfies BillingPublicMetadata,
    privateMetadata: {
      stripeCustomerId: asString(privateMetadata.stripeCustomerId),
      stripeSubscriptionId: asString(privateMetadata.stripeSubscriptionId),
      stripeSubscriptionItemId: asString(privateMetadata.stripeSubscriptionItemId),
      lastCheckoutSessionId: asString(privateMetadata.lastCheckoutSessionId),
    } satisfies BillingPrivateMetadata,
  };
}

export async function canCreateJob(userId: string) {
  const clerk = await clerkClient();
  const user = await clerk.users.getUser(userId);
  const state = getBillingState(user);
  const { plan, credits } = state.publicMetadata;

  const allowed =
    plan === "monthly" ||
    plan === "annual" ||
    (plan === "api_metered" && Boolean(state.privateMetadata.stripeSubscriptionItemId)) ||
    credits > 0;

  return {
    allowed,
    user,
    state,
  };
}

export async function grantCheckoutEntitlement(args: {
  userId: string;
  plan: CheckoutPlan;
  customerId?: string;
  sessionId?: string;
  subscriptionId?: string;
  subscriptionItemId?: string;
}) {
  const clerk = await clerkClient();
  const user = await clerk.users.getUser(args.userId);
  const state = getBillingState(user);
  const nextPublic: BillingPublicMetadata = { ...state.publicMetadata };
  const nextPrivate: BillingPrivateMetadata = { ...state.privateMetadata };

  if (args.customerId) {
    nextPrivate.stripeCustomerId = args.customerId;
  }

  if (args.sessionId) {
    nextPrivate.lastCheckoutSessionId = args.sessionId;
  }

  if (args.subscriptionId) {
    nextPrivate.stripeSubscriptionId = args.subscriptionId;
  }

  if (args.subscriptionItemId) {
    nextPrivate.stripeSubscriptionItemId = args.subscriptionItemId;
  }

  if (args.plan === "per_film") {
    nextPublic.plan = "per_film";
    nextPublic.credits += 1;
  } else {
    nextPublic.plan = args.plan;
  }

  await clerk.users.updateUserMetadata(args.userId, {
    publicMetadata: nextPublic,
    privateMetadata: nextPrivate,
  });

  return {
    publicMetadata: nextPublic,
    privateMetadata: nextPrivate,
  };
}

export async function revokeSubscriptionEntitlement(userId: string) {
  const clerk = await clerkClient();
  const user = await clerk.users.getUser(userId);
  const state = getBillingState(user);
  const nextPublic: BillingPublicMetadata = {
    ...state.publicMetadata,
    plan: state.publicMetadata.credits > 0 ? "per_film" : "free",
  };
  const nextPrivate: BillingPrivateMetadata = {
    ...state.privateMetadata,
    stripeSubscriptionId: undefined,
    stripeSubscriptionItemId: undefined,
  };

  await clerk.users.updateUserMetadata(userId, {
    publicMetadata: nextPublic,
    privateMetadata: nextPrivate,
  });

  return {
    publicMetadata: nextPublic,
    privateMetadata: nextPrivate,
  };
}

function clampUsageMinutes(value: unknown) {
  const parsed = Number(value);

  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 90;
  }

  return Math.min(24 * 60, Math.ceil(parsed));
}

export async function finalizeCompletedJob(userId: string, usageMinutes: unknown) {
  const clerk = await clerkClient();
  const user = await clerk.users.getUser(userId);
  const state = getBillingState(user);
  const nextPublic: BillingPublicMetadata = { ...state.publicMetadata };
  let metadataChanged = false;

  if (state.publicMetadata.credits > 0) {
    nextPublic.credits = Math.max(0, state.publicMetadata.credits - 1);
    if (state.publicMetadata.plan === "per_film" && nextPublic.credits === 0) {
      nextPublic.plan = "free";
    }
    metadataChanged = true;
  }

  if (state.publicMetadata.plan === "api_metered" && state.privateMetadata.stripeSubscriptionItemId) {
    const stripe = getStripe();
    await (stripe.subscriptionItems as unknown as {
      createUsageRecord: (
        subscriptionItemId: string,
        payload: {
          action: "increment";
          quantity: number;
          timestamp: number;
        },
      ) => Promise<unknown>;
    }).createUsageRecord(state.privateMetadata.stripeSubscriptionItemId, {
      action: "increment",
      quantity: clampUsageMinutes(usageMinutes),
      timestamp: Math.floor(Date.now() / 1000),
    });
  }

  if (metadataChanged) {
    await clerk.users.updateUserMetadata(userId, {
      publicMetadata: nextPublic,
      privateMetadata: state.privateMetadata,
    });
  }

  return {
    plan: state.publicMetadata.plan,
    creditsRemaining: metadataChanged ? nextPublic.credits : state.publicMetadata.credits,
  };
}

export function getBaseUrl(origin: string) {
  const configured =
    asString(process.env.NEXT_PUBLIC_APP_URL) ??
    asString(process.env.APP_URL) ??
    origin;

  return configured.startsWith("http") ? configured : `https://${configured}`;
}

export function getStripeCustomerId(value: unknown) {
  if (!value) {
    return undefined;
  }

  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "object" && "id" in value && typeof value.id === "string") {
    return value.id;
  }

  return undefined;
}

export function getSubscriptionItemId(
  items: Array<{ id: string; price?: { id?: string | null } | null }> | undefined,
  priceId: string | null | undefined,
) {
  if (!items?.length) {
    return undefined;
  }

  if (!priceId) {
    return items[0]?.id;
  }

  return items.find((item) => item.price?.id === priceId)?.id ?? items[0]?.id;
}
