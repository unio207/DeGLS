import { severityBand, type SeverityBand } from "@/lib/types";

/** Band boundaries, mirroring `severityBand()` in lib/types.ts — do not drift. */
export const BAND_STOPS = [0, 1, 5, 20, 50, 100] as const;

export const BAND_ORDER: SeverityBand[] = ["trace", "low", "moderate", "high", "severe"];

export const BAND_COLOR: Record<SeverityBand, string> = {
  trace: "var(--band-trace)",
  low: "var(--band-low)",
  moderate: "var(--band-moderate)",
  high: "var(--band-high)",
  severe: "var(--band-severe)",
};

export const BAND_LABEL: Record<SeverityBand, string> = {
  trace: "Trace",
  low: "Low",
  moderate: "Moderate",
  high: "High",
  severe: "Severe",
};

/**
 * Maps a 0–100 severity onto 0–1 of strip width, giving every band an equal
 * slice. A linear axis would compress trace/low/moderate — the range where
 * almost every real reading lands — into a fifth of the strip and leave most
 * of it permanently empty. Boundary ticks are drawn and labelled so the
 * non-linear axis is visible rather than implied.
 */
export function bandPosition(percent: number): number {
  const p = Math.min(100, Math.max(0, percent));
  const slice = 1 / (BAND_STOPS.length - 1);
  for (let i = 0; i < BAND_STOPS.length - 1; i += 1) {
    const lo = BAND_STOPS[i];
    const hi = BAND_STOPS[i + 1];
    if (p <= hi) {
      return (i + (p - lo) / (hi - lo)) * slice;
    }
  }
  return 1;
}

export function bandOf(percent: number): SeverityBand {
  return severityBand(percent);
}

/** One decimal below 10, whole numbers above — matches how raters write it down. */
export function formatSeverity(percent: number): string {
  const p = Math.min(100, Math.max(0, percent));
  return p < 10 ? p.toFixed(1) : Math.round(p).toString();
}

export function formatConfidence(confidence: number): string {
  return `${Math.round(Math.min(1, Math.max(0, confidence)) * 100)}%`;
}

/** Below this, the disease label is not worth acting on without a second look. */
export const LOW_CONFIDENCE = 0.6;
