"use client";

/**
 * Development fixtures.
 *
 * `/api/analyze` is a Python function that is built separately, so during UI
 * work there is nothing to talk to. Append `?mock=<case>` in development to
 * exercise a result state end to end. This never runs in a production build —
 * the real fetch in `analyze.ts` is always what ships.
 */

import type { AnalyzeResponse, DiseaseCode } from "@/lib/types";
import { DISEASE_LABELS } from "@/lib/types";
import { fileToDataUri, loadImage } from "@/lib/history";

export type MockCase =
  | "gls"
  | "nlb"
  | "rust"
  | "severe"
  | "lowconf"
  | "multi"
  | "no_leaf"
  | "unreliable_reading"
  | "invalid_image"
  | "file_too_large"
  | "internal";

export function mockCaseFrom(search: string): MockCase | null {
  if (process.env.NODE_ENV !== "development") return null;
  const value = new URLSearchParams(search).get("mock");
  if (!value) return null;
  const cases: MockCase[] = [
    "gls",
    "nlb",
    "rust",
    "severe",
    "lowconf",
    "multi",
    "no_leaf",
    "unreliable_reading",
    "invalid_image",
    "file_too_large",
    "internal",
  ];
  if (value === "1") return "gls";
  return cases.includes(value as MockCase) ? (value as MockCase) : "gls";
}

const PRESETS: Record<
  string,
  { code: DiseaseCode; confidence: number; percent: number; instances: number }
> = {
  gls: { code: "corn_gls", confidence: 0.91, percent: 7.4, instances: 1 },
  nlb: { code: "corn_nlb", confidence: 0.83, percent: 18.2, instances: 1 },
  rust: { code: "corn_rust", confidence: 0.95, percent: 2.3, instances: 1 },
  severe: { code: "corn_nlb", confidence: 0.88, percent: 61.7, instances: 1 },
  lowconf: { code: "corn_gls", confidence: 0.41, percent: 4.6, instances: 1 },
  multi: { code: "corn_rust", confidence: 0.79, percent: 12.9, instances: 3 },
};

export async function runMock(kase: MockCase, file: File): Promise<AnalyzeResponse> {
  await new Promise((r) => setTimeout(r, 2600));

  if (kase === "no_leaf") {
    return {
      ok: false,
      error: { code: "no_leaf_detected", message: "No corn leaf found in the image." },
    };
  }
  if (kase === "unreliable_reading") {
    return {
      ok: false,
      error: {
        code: "unreliable_reading",
        message:
          "This leaf reads as heavily diseased, which usually means widespread yellowing rather than lesions. Try a leaf with distinct spots on otherwise green tissue.",
      },
    };
  }
  if (kase === "invalid_image") {
    return { ok: false, error: { code: "invalid_image", message: "That file isn't a readable image." } };
  }
  if (kase === "file_too_large") {
    return { ok: false, error: { code: "file_too_large", message: "Image exceeds the 8 MB limit." } };
  }
  if (kase === "internal") {
    return { ok: false, error: { code: "internal", message: "Inference worker crashed." } };
  }

  const preset = PRESETS[kase] ?? PRESETS.gls;
  const overlay = await paintFakeOverlay(file);

  return {
    ok: true,
    disease: {
      code: preset.code,
      label: DISEASE_LABELS[preset.code],
      confidence: preset.confidence,
    },
    severity: {
      percent: preset.percent,
      lesion_px: Math.round(preset.percent * 1240),
      leaf_px: 124_000,
    },
    images: { overlay },
    meta: {
      processing_ms: 2610,
      instances_detected: preset.instances,
      multiple_leaves: preset.instances > 1,
      settings: { threshold: 0.35, tta: false, min_blob: 24 },
    },
  };
}

/** Paints plausible lesion blobs over the user's own photo so the compare wipe has something to show. */
async function paintFakeOverlay(file: File): Promise<string> {
  const src = await fileToDataUri(file);
  const img = await loadImage(src);
  const scale = Math.min(1, 900 / Math.max(img.width, img.height));
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return src;

  ctx.drawImage(img, 0, 0, w, h);
  ctx.globalAlpha = 0.55;
  ctx.fillStyle = "#e0338a";
  for (let i = 0; i < 26; i += 1) {
    const cx = w * (0.18 + Math.random() * 0.64);
    const cy = h * (0.12 + Math.random() * 0.76);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(Math.random() * Math.PI);
    ctx.beginPath();
    ctx.ellipse(0, 0, w * (0.01 + Math.random() * 0.05), h * 0.008, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "#25e07a";
  ctx.lineWidth = Math.max(2, w * 0.006);
  ctx.strokeRect(w * 0.08, h * 0.06, w * 0.84, h * 0.88);

  return canvas.toDataURL("image/jpeg", 0.85);
}
