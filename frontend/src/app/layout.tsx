import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EvidenceGraph AI | Accenture Innovation Challenge 2026",
  description:
    "Autonomous KPI anomaly investigation engine with typed evidence graph, GNN-powered driver ranking, PVM decomposition, and Confidence Gate verdict — built by Team HerForge.",
  keywords: "AI, KPI, investigation, evidence graph, GNN, Accenture, business intelligence",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
