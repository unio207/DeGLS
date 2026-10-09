"use client";

/**
 * Client-side photo shrinking.
 *
 * A recent phone shoots 12-48 MP; a single frame routinely lands between 8 and
 * 25 MB, over the function's MAX_UPLOAD_BYTES. Refusing it and asking someone
 * standing in a field to go change their camera resolution is a bad trade when
 * the browser can just re-encode the thing — and the upload is the slow half of
 * the wait on field signal anyway.
 *
 * Nothing is lost by doing it. YOLO letterboxes to 640x640 and GAUNet resizes to
 * 512x512, so anything past ~2000px on the long edge is discarded by the
 * pipeline regardless. The ladder below only ever engages for files that would
 * otherwise be rejected outright.
 *
 * TWO THINGS THAT WILL BITE YOU IF CHANGED:
 *
 * 1. `imageOrientation: "from-image"` is required. Canvas re-encoding strips
 *    EXIF, and cv2.imdecode on the server APPLIES EXIF orientation. Decode
 *    without baking rotation into the pixels and every portrait photo from an
 *    iPhone arrives at the model rotated 90 degrees, with no EXIF tag left to
 *    correct it.
 * 2. The output must be renamed .jpg. validate_upload() in api/analyze.py
 *    checks the filename extension against {png, jpg, jpeg}, so a re-encoded
 *    JPEG still called IMG_0001.heic is rejected by the server.
 */

/**
 * Every upload is capped at this on the long edge, not just oversized ones.
 *
 * This is not a bandwidth nicety. Production returned 500s on full-resolution
 * photos with "instance was killed because it ran out of available memory": a
 * 24.5 Mpx frame decodes to 73 MB server-side and peak RSS measured 1776 MB
 * against Vercel's 1024 MB limit. Files as small as 1.4 MB triggered it, since
 * what matters is pixel count, not bytes — so a byte-only threshold cannot fix
 * it and this has to apply unconditionally.
 *
 * api/analyze.py enforces the same cap independently (WORK_MAX_EDGE); doing it
 * here as well keeps the upload small over field signal.
 */
const MAX_EDGE = 2048;

/**
 * Progressively smaller re-encodes, first that fits wins. The 1024 floor stays
 * comfortably above YOLO's 640 input so the detector is never starved.
 */
const STEPS: ReadonlyArray<{ edge: number; quality: number }> = [
  { edge: MAX_EDGE, quality: 0.9 },
  { edge: 1600, quality: 0.85 },
  { edge: 1280, quality: 0.8 },
  { edge: 1024, quality: 0.75 },
];

function toJpegName(name: string): string {
  const base = name.replace(/\.[^./\\]+$/, "") || "photo";
  return `${base}.jpg`;
}

/**
 * Prepare `file` for upload: at most MAX_EDGE on the long side and `maxBytes`
 * on the wire.
 *
 * Returns the original file untouched when it already satisfies both, so small
 * photos are never needlessly re-encoded. Returns null if the image can't be
 * decoded (an HEIC on a browser without support, a corrupt file) or if even the
 * smallest step is still too large — the caller surfaces those as the ordinary
 * too-large error.
 */
export async function prepareForUpload(file: File, maxBytes: number): Promise<File | null> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    return null;
  }

  try {
    if (Math.max(bitmap.width, bitmap.height) <= MAX_EDGE && file.size <= maxBytes) {
      return file;
    }
    for (const { edge, quality } of STEPS) {
      const scale = Math.min(1, edge / Math.max(bitmap.width, bitmap.height));
      const w = Math.max(1, Math.round(bitmap.width * scale));
      const h = Math.max(1, Math.round(bitmap.height * scale));

      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(bitmap, 0, 0, w, h);

      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", quality),
      );
      if (!blob) return null;
      if (blob.size <= maxBytes) {
        return new File([blob], toJpegName(file.name), {
          type: "image/jpeg",
          lastModified: file.lastModified,
        });
      }
    }
    return null;
  } finally {
    bitmap.close();
  }
}
