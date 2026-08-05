"use client";

import type { AnalyzeErrorCode, AnalyzeResponse, LeafPoint, ScanInput } from "@/lib/types";

export interface AnalyzeOptions {
  file: File;
  input: ScanInput;
  /** Tap on the leaf, 0..1 in image space. Omitted, the server keeps today's behaviour. */
  point?: LeafPoint | null;
  /** 0–1 of the request body actually written to the wire. */
  onUploadProgress?: (fraction: number) => void;
  signal?: AbortSignal;
}

/**
 * POSTs the scan to the Python function.
 *
 * XMLHttpRequest instead of fetch purely for `upload.onprogress` — it is the
 * one part of the wait we can report truthfully, and on a field connection the
 * upload is usually the slow half.
 */
export function analyze({
  file,
  input,
  point,
  onUploadProgress,
  signal,
}: AnalyzeOptions): Promise<AnalyzeResponse> {
  return new Promise((resolve, reject) => {
    const body = new FormData();
    body.append("file", file);
    body.append("corn_hybrid", input.corn_hybrid);
    body.append("location", input.location);
    body.append("date", input.date);

    // `sam=1` travels with the point, not separately: SAM is off by default in
    // api/analyze.py (DEGLS_SAM) pending a memory check on Vercel, so the flag
    // is currently the only thing that makes the tap do anything at all.
    let url = "/api/analyze";
    if (point) {
      const q = new URLSearchParams({
        sam: "1",
        px: point.x.toFixed(4),
        py: point.y.toFixed(4),
      });
      url += `?${q}`;
    }

    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.responseType = "text";

    if (onUploadProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && e.total > 0) {
          onUploadProgress(Math.min(1, e.loaded / e.total));
        }
      };
      xhr.upload.onload = () => onUploadProgress(1);
    }

    xhr.onerror = () => reject(new NetworkError());
    xhr.ontimeout = () => reject(new NetworkError());
    xhr.onabort = () => reject(new DOMException("Scan cancelled.", "AbortError"));

    xhr.onload = () => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(xhr.responseText);
      } catch {
        resolve(failure(xhr.status === 413 ? "file_too_large" : "internal", serverText(xhr.status)));
        return;
      }
      if (isAnalyzeResponse(parsed)) {
        resolve(parsed);
        return;
      }
      resolve(failure("internal", serverText(xhr.status)));
    };

    if (signal) {
      if (signal.aborted) {
        reject(new DOMException("Scan cancelled.", "AbortError"));
        return;
      }
      signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }

    xhr.send(body);
  });
}

export class NetworkError extends Error {
  constructor() {
    super("The scan couldn't reach the server.");
    this.name = "NetworkError";
  }
}

const ERROR_CODES: AnalyzeErrorCode[] = [
  "no_leaf_detected",
  "unreliable_reading",
  "invalid_image",
  "file_too_large",
  "internal",
];

function isAnalyzeResponse(value: unknown): value is AnalyzeResponse {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  if (v.ok === true) return typeof v.disease === "object" && typeof v.severity === "object";
  if (v.ok === false) {
    const err = v.error as Record<string, unknown> | undefined;
    return !!err && ERROR_CODES.includes(err.code as AnalyzeErrorCode);
  }
  return false;
}

function failure(code: AnalyzeErrorCode, message: string): AnalyzeResponse {
  return { ok: false, error: { code, message } };
}

function serverText(status: number): string {
  if (status === 404) return "The analysis service isn't deployed at /api/analyze yet.";
  if (status === 413) return "The server rejected the photo as too large.";
  if (status >= 500) return `The analysis service returned ${status}.`;
  if (status === 0) return "The connection dropped mid-scan.";
  return `The analysis service returned an unexpected response (${status}).`;
}
