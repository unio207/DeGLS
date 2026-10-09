import type { Metadata, Viewport } from "next";
import { Archivo, IBM_Plex_Mono } from "next/font/google";

import { Toaster } from "@/components/ui/sonner";
import "./globals.css";

/**
 * Archivo carries the display voice — a grotesque drawn for signage, with a
 * tall x-height that holds up at arm's length in full sun. IBM Plex Mono is the
 * instrument voice: every field label, tick, coordinate and timestamp is set in
 * it, so a person can tell "something the app is telling you" from "something
 * you or the model measured" without reading a word.
 */
const archivo = Archivo({
  variable: "--font-archivo",
  subsets: ["latin"],
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "DeGLS — Corn leaf disease field scan",
  description:
    "Photograph a corn leaf to identify gray leaf spot, northern leaf blight or common rust and measure how much of the blade is lesioned.",
  applicationName: "DeGLS",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    title: "DeGLS",
    statusBarStyle: "default",
  },
  icons: {
    icon: [
      { url: "/icons/icon-32.png", sizes: "32x32", type: "image/png" },
      { url: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [{ url: "/icons/apple-touch-icon.png", sizes: "180x180", type: "image/png" }],
  },
  formatDetection: { telephone: false },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f9ef" },
    { media: "(prefers-color-scheme: dark)", color: "#101509" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${archivo.variable} ${plexMono.variable} h-full antialiased`}
    >
      <head>
        {/* Dark mode follows the device. Applied before paint so a phone taken
            out at dusk never flashes a white screen. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `try{var m=window.matchMedia('(prefers-color-scheme: dark)');var a=function(e){document.documentElement.classList.toggle('dark',e.matches)};a(m);m.addEventListener('change',a)}catch(e){}`,
          }}
        />
      </head>
      <body className="flex min-h-full flex-col">
        {children}
        <Toaster position="top-center" richColors closeButton />
      </body>
    </html>
  );
}
