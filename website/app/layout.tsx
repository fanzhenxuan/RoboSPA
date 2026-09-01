import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://fanzhenxuan.github.io/RoboSPA/"),
  title: "RoboSPA: Can VLA Models Go Beyond Simple Scenes and Short-Horizon Tasks?",
  description:
    "RoboSPA is a large-scale benchmark for fine-grained spatial reasoning and long-horizon procedural planning in Vision-Language-Action models.",
  openGraph: {
    title: "RoboSPA",
    description: "Beyond Simple Scenes and Short-Horizon Tasks",
    type: "website",
    images: [{ url: "https://fanzhenxuan.github.io/RoboSPA/og.png", width: 1731, height: 909, alt: "RoboSPA project preview" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "RoboSPA",
    description: "Beyond Simple Scenes and Short-Horizon Tasks",
    images: ["https://fanzhenxuan.github.io/RoboSPA/og.png"],
  },
  icons: {
    icon: "https://fanzhenxuan.github.io/RoboSPA/favicon.svg",
    shortcut: "https://fanzhenxuan.github.io/RoboSPA/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
