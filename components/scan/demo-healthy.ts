/**
 * TEMPORARY DEMO OVERRIDE — delete this file after the demo.
 *
 * When the hybrid field reads exactly "test", the result is presented as
 * healthy: no disease call, and the severity number replaced by
 * DEMO_HEALTHY_TEXT. The scan itself is untouched — the photo is uploaded, the
 * real request runs, the real overlay comes back, and errors take the normal
 * error path. Only the presentation is swapped: the overlay PNG has the red
 * lesion mask painted into it, so under the override the result shows the
 * segmented blade instead — the same picture without the lesion pass, kept on
 * the saved record so a reopened scan shows what the live one did.
 *
 * Exact match rather than substring: real hybrid names such as TEST123 must
 * read normally.
 *
 * Removal: delete this file, then drop the `presentAsHealthy` field and every
 * use of it TypeScript then flags (scan-app.tsx, result-panel.tsx and the
 * history/chat call sites).
 */
export function isDemoHealthyHybrid(hybrid: string): boolean {
  return hybrid.trim().toLowerCase() === "test";
}

export const DEMO_HEALTHY_TEXT = "Confidence too low to classify disease";
