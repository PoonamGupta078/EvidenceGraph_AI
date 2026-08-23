import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EvidenceGraph AI — Autonomous Anomaly Investigation",
  description:
    "Turn raw KPI divergences into auditable, persona-aware decisions using a typed Evidence Graph, Confidence Gate, and LLM narration.",
  keywords: "evidence graph, anomaly detection, root cause analysis, business intelligence",
  openGraph: {
    title: "EvidenceGraph AI",
    description: "Autonomous Anomaly Investigation Engine",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
