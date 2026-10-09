/**
 * TEMPORARY DEMO OVERRIDES — delete this file after the demo.
 *
 * Four reserved hybrid names, each an exact match so that real hybrid names
 * like TEST123 or P1197AM read normally:
 *
 *   test    presented as healthy — no disease call, and the severity number
 *           replaced by DEMO_HEALTHY_TEXT
 *   leaf1   Gray Leaf Spot
 *   leaf2   Northern Leaf Blight
 *   leaf3   Common Rust
 *
 * The scan itself is untouched in every case — the photo is uploaded, the real
 * request runs, the real overlay comes back, and errors take the normal error
 * path. Only what is shown changes.
 *
 * The two overrides differ in how much they swap:
 *
 *   `test` also replaces the picture. The overlay PNG has the red lesion mask
 *   painted into it, so a healthy-looking result cannot use it; the segmented
 *   blade is shown instead, and kept on the saved record so a reopened scan
 *   shows what the live one did.
 *
 *   `leaf1`-`leaf3` keep everything real except the name: the measured
 *   severity, the lesion overlay and the confidence are all the model's own
 *   output. Only `code` and `label` are substituted. That is why this one is
 *   applied once in showResult(), before the view and the record are built —
 *   the substituted disease is what gets stored, so history, the map and the
 *   assistant all read it without needing to know an override exists.
 *
 * Removal: delete this file, then drop the `presentAsHealthy` field and fix
 * every use TypeScript flags (scan-app.tsx, result-panel.tsx, and the
 * history/map/chat call sites).
 */
import { DISEASE_LABELS, type DiseaseCode, type DiseaseResult } from "@/lib/types";

export function isDemoHealthyHybrid(hybrid: string): boolean {
  return hybrid.trim().toLowerCase() === "test";
}

export const DEMO_HEALTHY_TEXT = "Confidence too low to classify disease";

const DEMO_DISEASE_HYBRIDS: Record<string, DiseaseCode> = {
  leaf1: "corn_gls",
  leaf2: "corn_nlb",
  leaf3: "corn_rust",
};

/**
 * The forced call for a reserved hybrid, or null to leave the reading alone.
 * Confidence is deliberately not touched — it is the model's real number, and
 * a fabricated one would be the easiest part of this to spot.
 */
export function applyDemoDisease(hybrid: string, disease: DiseaseResult): DiseaseResult {
  const code = DEMO_DISEASE_HYBRIDS[hybrid.trim().toLowerCase()];
  return code ? { ...disease, code, label: DISEASE_LABELS[code] } : disease;
}
