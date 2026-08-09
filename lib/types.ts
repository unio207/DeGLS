/**
 * Shared contract between the Python inference function (`api/analyze.py`),
 * the chat route (`app/api/chat/route.ts`), and the UI.
 *
 * This file is the source of truth. If the Python side changes its response
 * shape, change it here first.
 */

import type { UIMessage } from "ai";

/** Class codes emitted by the YOLOv8s-seg model (nc=3). */
export type DiseaseCode = "corn_gls" | "corn_nlb" | "corn_rust";

/** Human-readable labels, matching what the Python function returns. */
export const DISEASE_LABELS: Record<DiseaseCode, string> = {
  corn_gls: "Gray Leaf Spot",
  corn_nlb: "Northern Leaf Blight",
  corn_rust: "Common Rust",
};

export interface DiseaseResult {
  code: DiseaseCode;
  label: string;
  /** 0-1, from the highest-confidence YOLO instance. */
  confidence: number;
}

export interface SeverityResult {
  /** Lesion pixels as a percentage of leaf pixels, 0-100. */
  percent: number;
  lesion_px: number;
  leaf_px: number;
}

export interface AnalyzeMeta {
  processing_ms: number;
  instances_detected: number;
  /** True when >1 leaf was detected; severity reflects only the top instance. */
  multiple_leaves: boolean;
  /**
   * Which path produced the leaf mask. Diagnostics only — never shown to a
   * grower, who has no way to act on it. Optional here because history records
   * and the mock fixtures predate it.
   */
  mask_source?: "yolo" | "sam_tap" | "sam_centre" | "yolo_sam_failed";
  settings: {
    threshold: number;
    tta: boolean;
    min_blob: number;
  };
}

export interface AnalyzeSuccess {
  ok: true;
  disease: DiseaseResult;
  severity: SeverityResult;
  /**
   * Data URIs, since the serverless filesystem is ephemeral.
   *
   * `overlay` is the flattened picture: leaf on black with the red lesion mask
   * painted in. `leaf_cutout` is the leaf mask on its own — a transparent PNG,
   * black outside the blade and 10% black over it — so the segmentation can be
   * shown over the photo without the lesion pass. Optional because history
   * records and the dev fixtures predate it.
   */
  images: { overlay: string; leaf_cutout?: string };
  meta: AnalyzeMeta;
}

export type AnalyzeErrorCode =
  | "no_leaf_detected"
  | "unreliable_reading"
  | "invalid_image"
  | "file_too_large"
  | "internal";

/**
 * Debugging detail carried alongside a failure. Shown small and muted on the
 * error screen so a grower in a field can read the code down the phone, and so
 * the reason names the stage that actually broke rather than the UI bucket it
 * fell into. Optional because a deployment older than this field won't send it.
 */
export interface AnalyzeDiag {
  /** Short, stable, greppable: `DG-<area>-<fault>`, e.g. "DG-FN-KILLED". */
  code: string;
  /** One technical line: what failed and where. */
  reason: string;
  /** `x-vercel-id` off the response, when the platform supplied one. */
  requestId?: string;
}

export interface AnalyzeFailure {
  ok: false;
  error: { code: AnalyzeErrorCode; message: string; diag?: AnalyzeDiag };
}

export type AnalyzeResponse = AnalyzeSuccess | AnalyzeFailure;

/**
 * Where the grower tapped the leaf, in IMAGE space: x = column / imageWidth,
 * y = row / imageHeight, both 0..1. Normalised rather than in pixels because
 * prepareForUpload may re-encode the photo at a smaller size between the tap
 * and the upload, and a fraction survives that untouched.
 */
export interface LeafPoint {
  x: number;
  y: number;
}

/** Form fields collected alongside the image upload. */
export interface ScanInput {
  corn_hybrid: string;
  location: string;
  /** ISO date, YYYY-MM-DD. */
  date: string;
}

/** A completed scan as persisted to IndexedDB. Local-only, never sent to a server. */
export interface ScanRecord extends ScanInput {
  id: string;
  /** Epoch ms, when the scan was run. */
  created_at: number;
  disease: DiseaseResult;
  severity: SeverityResult;
  /** Downscaled JPEG data URI — full-size overlays would blow out IndexedDB. */
  thumbnail: string;
  overlay: string;
  /**
   * The segmented blade with no lesion marks, flattened to one JPEG at save
   * time (see flattenCutout). Optional: records written before this field
   * existed do not have it, and a reopened scan without one simply shows no
   * segmentation, exactly as it did then.
   */
  segmented?: string;
}

/**
 * Severity bands, used for the result label and gauge color.
 *
 * Cut points follow conventional foliar-disease severity reporting for corn.
 * (Engineering note, not for display: these were chosen by convention rather
 * than fit to data, since the training set is no longer available.)
 */
export type SeverityBand = "trace" | "low" | "moderate" | "high" | "severe";

export function severityBand(percent: number): SeverityBand {
  if (percent < 1) return "trace";
  if (percent < 5) return "low";
  if (percent < 20) return "moderate";
  if (percent < 50) return "high";
  return "severe";
}

/** Context handed to the chat model so answers reference the actual scan. */
export interface DiagnosisContext extends ScanInput {
  disease_label: string;
  disease_code: DiseaseCode;
  severity_percent: number;
  confidence: number;
  /**
   * The scan produced no usable classification, so the assistant must not name
   * a disease or quote a severity. Currently set only by the temporary demo
   * override (see components/scan/demo-healthy.ts).
   */
  unclassified?: boolean;
}

export interface ChatCitation {
  /** Source document title, e.g. "Gray Leaf Spot of Corn (Crop Protection Network)". */
  title: string;
  /** Public URL of the source publication, when known. */
  url?: string;
}

/**
 * Per-message metadata streamed from the chat route.
 *
 * Citations are attached at the `start` part rather than parsed out of the
 * model's text: retrieval has already happened by then, so the UI can render
 * the source list immediately and it cannot drift from what was actually
 * retrieved. `citations` is empty when nothing was retrieved, and the UI then
 * renders no source area at all.
 */
export interface ChatMessageMetadata {
  citations?: ChatCitation[];
  /** False when the answer came from the diagnosis alone (no corpus hits). */
  grounded?: boolean;
}

export type DiagnosisUIMessage = UIMessage<ChatMessageMetadata>;

/**
 * A chat thread, keyed to the scan it is about.
 *
 * Held in its own IndexedDB store rather than on `ScanRecord` because the
 * record carries a full-size overlay data URI: rewriting all of it on every
 * chat message would be a multi-megabyte write per turn.
 */
export interface ConversationRecord {
  /** `ScanRecord.id` of the scan this conversation belongs to. */
  scan_id: string;
  /** Epoch ms of the last message saved. */
  updated_at: number;
  messages: DiagnosisUIMessage[];
}
