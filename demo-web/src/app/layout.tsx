import { ClerkProvider } from "@clerk/nextjs";
import { Analytics } from "@vercel/analytics/react";
import type { Metadata } from "next";
import { DM_Serif_Display, DM_Sans, IBM_Plex_Mono } from "next/font/google";
import type { ReactNode } from "react";
import "./globals.css";

const dmSerifDisplay = DM_Serif_Display({
  subsets: ["latin"],
  variable: "--font-playfair",
  weight: ["400"],
  display: "swap",
});

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-geist",
  weight: ["400", "500", "600"],
  display: "swap",
});

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Visual Narrator — Audio Description in Minutes",
  description:
    "Upload a film or paste a URL. Get a broadcast-ready audio description track — MP3, SRT, and compliance report — in about 20 minutes. $39.",
  metadataBase: new URL("https://app.vnpoverview.com"),
  openGraph: {
    title: "Visual Narrator — Audio Description in Minutes",
    description: "Broadcast-ready AD tracks for indie filmmakers, EdTech platforms, and AD studios.",
    type: "website",
  },
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  const body = (
    <html
      className={`${dmSerifDisplay.variable} ${dmSans.variable} ${ibmPlexMono.variable}`}
      data-theme="dark"
      lang="en"
    >
      <body>
        {/* Film grain — fixed overlay, GPU-isolated */}
        <div aria-hidden="true" className="vn-grain-layer" />
        {children}
        <Analytics />
      </body>
    </html>
  );

  if (!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY) {
    return body;
  }

  return (
    <ClerkProvider publishableKey={process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}>
      {body}
    </ClerkProvider>
  );
}
