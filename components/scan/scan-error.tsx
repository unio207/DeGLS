"use client";

import {
  CameraIcon,
  CheckIcon,
  CopyIcon,
  FileWarningIcon,
  LeafIcon,
  RotateCwIcon,
  ScaleIcon,
  ServerCrashIcon,
  WifiOffIcon,
} from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import type { AnalyzeDiag, AnalyzeErrorCode } from "@/lib/types";

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
  // Now reachable two ways: nothing leaf-like in the photo at all, or a marker
  // sitting on soil or shadow beside the blade. Moving the marker is the
  // cheaper fix and needs no new photo, so it leads — see onAdjust below, which
  // replaces the primary action for this code only.
  //
  // Worded around the MARKER, not a tap: CaptureCard now auto-places it at the
  // centre of every photo, so a grower can land here without having deliberately
  // tapped anything, and "tap again" would be describing something they never did.
  no_leaf_detected: {
    icon: LeafIcon,
    title: "No leaf found where the marker is",
    body: "Nothing measurable as a corn leaf was found at that spot. Either the marker isn't on the blade, or the photo hasn't got a leaf the detector can read.",
    tips: [
      "Drag the marker well inside the blade — the middle of it, not an edge, a shadow or the soil behind it.",
      "If the marker was already on the leaf, retake: fill more of the frame, blade within about a foot of the lens.",
      "Hold the leaf flat and shoot square to it, not at a steep angle.",
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

function execCommandCopy(text: string): boolean {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.cssText = "position:fixed;top:0;left:0;opacity:0";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  ta.remove();
  return ok;
}

/**
 * The code, big enough to hit with a thumb and copyable in one tap — in a field
 * the alternative is transcribing it onto paper. What lands on the clipboard is
 * everything needed to find the invocation in the Vercel log, not just the code.
 *
 * The icon swap is the only thing that changes on copy, and it is aria-hidden,
 * so the surrounding role="alert" region has nothing new to re-announce.
 */
function CopyCode({ diag, uiCode }: { diag: AnalyzeDiag; uiCode: ScanErrorCode }) {
  const [copied, setCopied] = useState(false);

  async function copyAll() {
    const lines = [
      diag.code,
      `ui: ${uiCode}`,
      `reason: ${diag.reason}`,
      diag.requestId ? `vercel-id: ${diag.requestId}` : null,
      `at: ${new Date().toISOString()}`,
    ].filter(Boolean);
    const text = lines.join("\n");
    let ok = false;
    try {
      await navigator.clipboard.writeText(text);
      ok = true;
    } catch {
      // Safari on iOS refuses the async clipboard often enough that the field
      // demo can't rely on it; the deprecated path still works from a tap.
      ok = execCommandCopy(text);
    }
    // Only claim success when the write actually happened — otherwise the code
    // is still on screen to read out, which is the fallback that never fails.
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    }
  }

  return (
    <button
      type="button"
      onClick={copyAll}
      aria-label={`Copy error code ${diag.code} and details`}
      className="text-foreground hover:bg-secondary flex items-center gap-2 rounded-md border px-2 py-1 font-mono text-[0.8125rem] font-semibold tracking-tight"
    >
      {diag.code}
      {copied ? (
        <CheckIcon aria-hidden className="size-3.5 shrink-0" />
      ) : (
        <CopyIcon aria-hidden className="size-3.5 shrink-0" />
      )}
    </button>
  );
}

export function ScanError({
  code,
  detail,
  diag,
  onRetake,
  onRetry,
  onAdjust,
}: {
  code: ScanErrorCode;
  detail?: string;
  /** Debug code + technical reason, when the failure carried one. */
  diag?: AnalyzeDiag;
  onRetake: () => void;
  onRetry: () => void;
  /** Back to the photo with the tap marker still on it, to move it. */
  onAdjust?: () => void;
}) {
  const copy = COPY[code] ?? COPY.internal;
  const Icon = copy.icon;
  // The diag code shows on every failure including a missed leaf: that one is
  // the most common thing to hit in a field, and DG-LEAF-NONE (detector found
  // nothing) vs DG-LEAF-TAP (found a leaf, marker missed it) vs
  // DG-LEAF-IMPLAUSIBLE (rejected on pixels) is the most useful thing a photo
  // of this screen can tell us. The server *message* still doesn't show there,
  // because for a missed leaf it only restates the title.
  const showDetail = diag || (detail && code !== "no_leaf_detected");

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
          <div className="text-muted-foreground space-y-1 border-l-2 pl-3 font-mono text-[0.75rem] leading-snug">
            {diag ? (
              <>
                <CopyCode diag={diag} uiCode={code} />
                <p>{diag.reason}</p>
                {diag.requestId && <p className="break-all">req {diag.requestId}</p>}
              </>
            ) : (
              <p>{detail}</p>
            )}
          </div>
        )}

        <div className="grid gap-2">
          {onAdjust && code === "no_leaf_detected" ? (
            // Retrying unchanged would send the identical point and fail the
            // identical way, so for a missed tap the way out is the photo, not
            // the network.
            <>
              <Button
                onClick={onAdjust}
                className="h-14 w-full gap-2.5 rounded-xl text-base font-semibold md:h-15 md:text-lg"
              >
                <LeafIcon aria-hidden className="size-5 md:size-6" />
                Move the marker
              </Button>
              <Button
                variant="outline"
                onClick={onRetake}
                className="h-12 w-full rounded-xl text-[0.9375rem] md:h-13 md:text-base"
              >
                Take another photo
              </Button>
            </>
          ) : copy.primary === "retake" ? (
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
          {!(onAdjust && code === "no_leaf_detected") && (
            <Button
              variant="outline"
              onClick={copy.primary === "retake" ? onRetry : onRetake}
              className="h-12 w-full rounded-xl text-[0.9375rem] md:h-13 md:text-base"
            >
              {copy.primary === "retake" ? "Send this photo again anyway" : "Take another photo"}
            </Button>
          )}
        </div>
      </div>
    </section>
  );
}
