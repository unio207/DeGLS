"use client";

import {
  CameraIcon,
  FileWarningIcon,
  LeafIcon,
  RotateCwIcon,
  ScaleIcon,
  ServerCrashIcon,
  WifiOffIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import type { AnalyzeErrorCode } from "@/lib/types";

export type ScanErrorCode = AnalyzeErrorCode | "network";

interface Copy {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  body: string;
  tips: string[];
  primary: "retake" | "retry";
}

/**
 * Every failure the API can return, written as instructions rather than as an
 * apology. `no_leaf_detected` comes back with HTTP 200 and is by far the most
 * common outcome, so it gets the fullest guidance — it is a framing problem,
 * not a crash, and the old UI logged it to the console and showed nothing.
 */
const COPY: Record<ScanErrorCode, Copy> = {
  no_leaf_detected: {
    icon: LeafIcon,
    title: "No leaf found in that photo",
    body: "The detector didn't find a corn leaf it could measure. That's almost always framing rather than the plant.",
    tips: [
      "Fill more of the frame — get the blade within about a foot of the lens.",
      "Hold the leaf flat and shoot square to it, not at a steep angle.",
      "Step so your own shadow is off the leaf, or shade it evenly with your body.",
      "One leaf at a time against a plain background, not a wall of canopy.",
    ],
    primary: "retake",
  },
  // TEMPORARY DEMO GUARD - paired with DEGLS_SEVERITY_GUARD in api/analyze.py.
  // Fires when severity exceeds the plausibility ceiling, which in practice
  // means a chlorotic leaf read as heavily diseased. Remove with the guard.
  unreliable_reading: {
    icon: LeafIcon,
    title: "That reading doesn't look trustworthy",
    body: "The leaf measured as heavily diseased. Uniform yellowing — from nitrogen, drought, or a leaf that's simply old — gets read as lesion tissue, so a very high number is usually that rather than an outbreak.",
    tips: [
      "Look at the leaf: scattered spots on green tissue is disease, an evenly pale or yellow blade usually isn't.",
      "Pick a leaf with distinct lesions and green tissue still around them.",
      "Lower leaves yellow naturally as the plant matures — try one higher up the stalk.",
    ],
    primary: "retake",
  },
  invalid_image: {
    icon: FileWarningIcon,
    title: "That file couldn't be read",
    body: "The server couldn't open the photo as an image.",
    tips: [
      "Take a fresh photo with the camera button rather than sharing a screenshot or a document.",
      "If it came from another app, save it as a JPEG or PNG first.",
    ],
    primary: "retake",
  },
  file_too_large: {
    icon: ScaleIcon,
    title: "That photo is too big to send",
    body: "The upload exceeded the size the analysis service accepts.",
    tips: [
      "Drop your camera to a lower resolution and shoot again.",
      "If the photo came from a library, pick a smaller copy.",
    ],
    primary: "retake",
  },
  internal: {
    icon: ServerCrashIcon,
    title: "The analysis failed on the server",
    body: "Something broke while the model was running. Your photo and field note are still here.",
    tips: ["Try again — most of these clear on a second attempt."],
    primary: "retry",
  },
  network: {
    icon: WifiOffIcon,
    title: "Couldn't reach the analysis service",
    body: "The scan didn't make it out. Field signal is the usual cause.",
    tips: [
      "Move somewhere with a bar or two and try again.",
      "Your photo and field note are kept, so nothing needs re-entering.",
    ],
    primary: "retry",
  },
};

export function ScanError({
  code,
  detail,
  onRetake,
  onRetry,
}: {
  code: ScanErrorCode;
  detail?: string;
  onRetake: () => void;
  onRetry: () => void;
}) {
  const copy = COPY[code] ?? COPY.internal;
  const Icon = copy.icon;
  // For a missed leaf the server message only restates the title, so it is
  // dropped; for the rarer codes it is the only clue about what actually broke.
  const showDetail = detail && code !== "no_leaf_detected";

  return (
    <section
      role="alert"
      aria-live="assertive"
      className="bg-card animate-rise overflow-hidden rounded-2xl border"
    >
      <div className="bg-secondary text-secondary-foreground flex items-start gap-3 px-4 py-3.5 md:px-5 md:py-4">
        <Icon className="mt-0.5 size-5 shrink-0" />
        <h2 className="font-display text-lg leading-tight font-bold md:text-xl">{copy.title}</h2>
      </div>

      <div className="space-y-4 p-4 md:space-y-5 md:p-5">
        <p className="text-[0.9375rem] leading-relaxed md:text-base">{copy.body}</p>

        <ul className="space-y-2">
          {copy.tips.map((tip) => (
            <li key={tip} className="flex gap-2.5 text-[0.9375rem] leading-snug md:text-base">
              <span className="bg-primary mt-2 size-1.5 shrink-0 rounded-full" aria-hidden />
              <span>{tip}</span>
            </li>
          ))}
        </ul>

        {showDetail && (
          <p className="text-muted-foreground border-l-2 pl-3 font-mono text-[0.75rem] leading-snug">
            {detail}
          </p>
        )}

        <div className="grid gap-2">
          {copy.primary === "retake" ? (
            <Button
              onClick={onRetake}
              className="h-14 w-full gap-2.5 rounded-xl text-base font-semibold md:h-15 md:text-lg"
            >
              <CameraIcon aria-hidden className="size-5 md:size-6" />
              Take another photo
            </Button>
          ) : (
            <Button
              onClick={onRetry}
              className="h-14 w-full gap-2.5 rounded-xl text-base font-semibold md:h-15 md:text-lg"
            >
              <RotateCwIcon aria-hidden className="size-5 md:size-6" />
              Try again
            </Button>
          )}
          <Button
            variant="outline"
            onClick={copy.primary === "retake" ? onRetry : onRetake}
            className="h-12 w-full rounded-xl text-[0.9375rem] md:h-13 md:text-base"
          >
            {copy.primary === "retake" ? "Send this photo again anyway" : "Take another photo"}
          </Button>
        </div>
      </div>
    </section>
  );
}
