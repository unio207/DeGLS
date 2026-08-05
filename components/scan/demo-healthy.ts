/**
 * TEMPORARY DEMO OVERRIDE — delete this file after the demo.
 *
 * When the hybrid field reads exactly "test", the result is presented as
 * healthy: no disease call, and the severity number replaced by
 * DEMO_HEALTHY_TEXT. The scan itself is untouched — the photo is uploaded, the
 * real request runs, the real overlay comes back, and errors take the normal
 * error path. Only the diagnosis readout is swapped.
 *
 * Exact match rather than substring: real hybrid names such as TEST123 must
 * read normally.
 *
 * Removal: delete this file, then drop the `presentAsHealthy` field and its
 * three uses in scan-app.tsx and result-panel.tsx (TypeScript will point at
 * every one of them).
 */
export function isDemoHealthyHybrid(hybrid: string): boolean {
  return hybrid.trim().toLowerCase() === "test";
}

export const DEMO_HEALTHY_TEXT = "Confidence too low to classify disease";
