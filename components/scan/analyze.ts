"use client";

import type {
  AnalyzeDiag,
  AnalyzeErrorCode,
  AnalyzeResponse,
  LeafPoint,
  ScanInput,
} from "@/lib/types";

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

    xhr.onerror = () =>
      reject(
        new NetworkError({
          code: "DG-NET-LOST",
          reason: `XHR error event at readyState ${xhr.readyState}, status ${xhr.status} — the request never completed (offline, DNS, or the connection dropped mid-upload).`,
        }),
      );
    xhr.ontimeout = () =>
      reject(
        new NetworkError({
          code: "DG-NET-TIMEOUT",
          reason: `XHR timed out client-side after ${xhr.timeout} ms — nothing came back from /api/analyze.`,
        }),
      );
    xhr.onabort = () => reject(new DOMException("Scan cancelled.", "AbortError"));

    xhr.onload = () => {
      // Same-origin, so every response header is readable. `x-vercel-id` is the
      // request id in Vercel's logs, and `x-vercel-error` is set by the platform
      // on failures our handler never got to write (verified on the deployment:
      // a 5 MB body returns 413 + FUNCTION_PAYLOAD_TOO_LARGE as text/plain).
      const requestId = xhr.getResponseHeader("x-vercel-id") ?? undefined;
      const platformError = xhr.getResponseHeader("x-vercel-error") ?? undefined;

      let parsed: unknown;
      try {
        parsed = JSON.parse(xhr.responseText);
      } catch {
        resolve(nonJsonFailure(xhr.status, xhr.responseText, platformError, requestId, !!point));
        return;
      }
      if (isAnalyzeResponse(parsed)) {
        // Stamp the request id onto whatever the server said, so one code and
        // one id identify the invocation in the Vercel log.
        if (parsed.ok === false && requestId) {
          parsed.error.diag = {
            code: parsed.error.diag?.code ?? "DG-SRV-UNSPECIFIED",
            reason: parsed.error.diag?.reason ?? parsed.error.message,
            requestId,
          };
        }
        resolve(parsed);
        return;
      }
      resolve(
        failure("internal", serverText(xhr.status), {
          code: "DG-API-SHAPE",
          reason: `HTTP ${xhr.status} returned JSON that is not an AnalyzeResponse. ${preview(xhr.responseText)}`,
          requestId,
        }),
      );
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
  readonly diag: AnalyzeDiag;

  constructor(diag: AnalyzeDiag) {
    super("The scan couldn't reach the server.");
    this.name = "NetworkError";
    this.diag = diag;
  }
}

/**
 * The response body was not JSON, which means our handler never wrote it: the
 * platform answered instead, or the function died before replying. These are
 * the failures that used to collapse into one `internal`.
 */
function nonJsonFailure(
  status: number,
  body: string,
  platformError: string | undefined,
  requestId: string | undefined,
  samRequested: boolean,
): AnalyzeResponse {
  const seen = platformError ? `x-vercel-error: ${platformError}` : preview(body);

  if (status === 413) {
    return failure("file_too_large", serverText(status), {
      code: "DG-EDGE-413",
      reason: `Vercel rejected the body at the edge (4.5 MB request limit) before api/analyze.py ran, so the function never saw the upload. ${seen}`,
      requestId,
    });
  }
  if (status === 504) {
    return failure("internal", serverText(status), {
      code: "DG-FN-TIMEOUT",
      reason: `HTTP 504 — the function passed the 60s maxDuration in vercel.json. ${seen}`,
      requestId,
    });
  }
  if (status === 500 || status === 502) {
    // Nothing was returned, so this is inference, not a report: a function
    // killed for exceeding Vercel's 1024 MB cannot write a response, and the
    // SAM path is what pushes it there. A plain crash looks identical from here.
    return failure("internal", serverText(status), {
      code: "DG-FN-KILLED",
      reason: `HTTP ${status} with a non-JSON body — api/analyze.py died before replying. Usually the 1024 MB memory cap (${samRequested ? "sam=1 was requested, which is the path that hits it" : "sam was not requested"}), otherwise a crash at import. ${seen}`,
      requestId,
    });
  }
  if (status === 404) {
    return failure("internal", serverText(status), {
      code: "DG-API-404",
      reason: `POST /api/analyze returned 404 — the Python function is not deployed here. ${seen}`,
      requestId,
    });
  }
  if (status === 0) {
    return failure("internal", serverText(status), {
      code: "DG-NET-LOST",
      reason: "Load event with status 0 — the connection ended without a response.",
      requestId,
    });
  }
  return failure("internal", serverText(status), {
    code: `DG-HTTP-${status}`,
    reason: `Unexpected HTTP ${status} with a non-JSON body. ${seen}`,
    requestId,
  });
}

/** First line-ish of a response body, for the reason string. */
function preview(body: string): string {
  const flat = body.replace(/\s+/g, " ").trim();
  return flat ? `body: ${flat.slice(0, 140)}` : "body was empty";
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

function failure(code: AnalyzeErrorCode, message: string, diag: AnalyzeDiag): AnalyzeResponse {
  return { ok: false, error: { code, message, diag } };
}

function serverText(status: number): string {
  if (status === 404) return "The analysis service isn't deployed at /api/analyze yet.";
  if (status === 413) return "The server rejected the photo as too large.";
  if (status >= 500) return `The analysis service returned ${status}.`;
  if (status === 0) return "The connection dropped mid-scan.";
  return `The analysis service returned an unexpected response (${status}).`;
}
