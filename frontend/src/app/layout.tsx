import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Burn Boundary — microgravity fire safety for exploration atmospheres",
  description: "Which materials go from safe to dangerous when the atmosphere changes from ISS to Moon base — built on NASA PSI combustion data.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased font-sans">{children}</body>
    </html>
  );
}
