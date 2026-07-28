/**
 * Shared contract between the Python inference function (`api/analyze.py`),
 * the chat route (`app/api/chat/route.ts`), and the UI.
 *
 * This file is the source of truth. If the Python side changes its response
 * shape, change it here first.
 */

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
  /** Data URIs, since the serverless filesystem is ephemeral. */
  images: { overlay: string };
  meta: AnalyzeMeta;
}

export type AnalyzeErrorCode =
  | "no_leaf_detected"
  | "invalid_image"
  | "file_too_large"
  | "internal";

export interface AnalyzeFailure {
  ok: false;
  error: { code: AnalyzeErrorCode; message: string };
}

export type AnalyzeResponse = AnalyzeSuccess | AnalyzeFailure;

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
}

export interface ChatCitation {
  /** Source document title, e.g. "Gray Leaf Spot of Corn (Crop Protection Network)". */
  title: string;
  /** Public URL of the source publication, when known. */
  url?: string;
}
