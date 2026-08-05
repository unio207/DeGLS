"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CheckIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

/**
 * Crop the photo down to the leaf before analysing it.
 *
 * Measured on a field photo where the blade filled about a quarter of the
 * frame: cropping in dropped the detected region from 7 disconnected pieces to
 * 1, and severity from 14.1% to 9.0% as background weeds and soil stopped being
 * counted. The detector cannot separate a leaf from clutter it is touching, so
 * excluding the clutter at capture time is the one lever that reliably works.
 *
 * Deliberately a rectangle with corner handles rather than a freeform lasso: a
 * corn blade is close to a rectangle in frame, and this has to be usable with a
 * thumb in a field.
 */

const MIN_FRACTION = 0.08; // smallest crop, as a fraction of the image
type Handle = "nw" | "ne" | "sw" | "se" | "move" | null;

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Rect in 0..1 image space, so it survives the preview being any size. */
const INITIAL: Rect = { x: 0.18, y: 0.08, w: 0.64, h: 0.84 };

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

export function CropDialog({
  open,
  onOpenChange,
  file,
  onCropped,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  file: File | null;
  onCropped: (cropped: File) => void;
}) {
  const src = useMemo(() => (file && open ? URL.createObjectURL(file) : null), [file, open]);
  useEffect(() => {
    if (!src) return;
    return () => URL.revokeObjectURL(src);
  }, [src]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {/* Wider on a tablet so the crop rectangle is dragged at closer to the
          photo's own scale — a 28px corner handle on a 300px-wide preview is a
          much coarser instrument than the same handle on 600px. */}
      <DialogContent className="max-w-[min(96vw,32rem)] gap-3 p-4 md:max-w-[min(92vw,44rem)] md:gap-4 md:p-5">
        <DialogHeader className="text-left">
          <DialogTitle className="font-display text-lg font-extrabold tracking-tight md:text-xl">
            Crop to the leaf
          </DialogTitle>
          <DialogDescription className="text-[0.8125rem] md:text-sm">
            Drag the corners so the box holds the blade and as little else as possible. Background
            weeds and soil get measured otherwise.
          </DialogDescription>
        </DialogHeader>
        {/* Mounted only while open, so the crop rectangle resets to INITIAL on
            each use without an effect writing state. */}
        {src && file && (
          <CropBody
            src={src}
            file={file}
            onCropped={onCropped}
            onClose={() => onOpenChange(false)}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

function CropBody({
  src,
  file,
  onCropped,
  onClose,
}: {
  src: string;
  file: File;
  onCropped: (cropped: File) => void;
  onClose: () => void;
}) {
  const [rect, setRect] = useState<Rect>(INITIAL);
  const [busy, setBusy] = useState(false);
  const frameRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ handle: Handle; startX: number; startY: number; start: Rect } | null>(
    null,
  );

  const pointerFraction = useCallback((e: React.PointerEvent) => {
    const box = frameRef.current?.getBoundingClientRect();
    if (!box) return { fx: 0, fy: 0 };
    return { fx: (e.clientX - box.left) / box.width, fy: (e.clientY - box.top) / box.height };
  }, []);

  function beginDrag(e: React.PointerEvent, handle: Handle) {
    e.preventDefault();
    e.stopPropagation();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const { fx, fy } = pointerFraction(e);
    dragRef.current = { handle, startX: fx, startY: fy, start: { ...rect } };
  }

  function onMove(e: React.PointerEvent) {
    const drag = dragRef.current;
    if (!drag) return;
    const { fx, fy } = pointerFraction(e);
    const dx = fx - drag.startX;
    const dy = fy - drag.startY;
    const s = drag.start;

    if (drag.handle === "move") {
      setRect({
        ...s,
        x: clamp01(Math.min(s.x + dx, 1 - s.w)),
        y: clamp01(Math.min(s.y + dy, 1 - s.h)),
      });
      return;
    }

    // Corner drags move one corner and keep the opposite one pinned.
    let { x, y, w, h } = s;
    if (drag.handle === "nw") {
      const nx = clamp01(Math.min(s.x + dx, s.x + s.w - MIN_FRACTION));
      const ny = clamp01(Math.min(s.y + dy, s.y + s.h - MIN_FRACTION));
      w = s.x + s.w - nx;
      h = s.y + s.h - ny;
      x = nx;
      y = ny;
    } else if (drag.handle === "ne") {
      const nw = Math.max(MIN_FRACTION, Math.min(s.w + dx, 1 - s.x));
      const ny = clamp01(Math.min(s.y + dy, s.y + s.h - MIN_FRACTION));
      h = s.y + s.h - ny;
      w = nw;
      y = ny;
    } else if (drag.handle === "sw") {
      const nx = clamp01(Math.min(s.x + dx, s.x + s.w - MIN_FRACTION));
      w = s.x + s.w - nx;
      h = Math.max(MIN_FRACTION, Math.min(s.h + dy, 1 - s.y));
      x = nx;
    } else if (drag.handle === "se") {
      w = Math.max(MIN_FRACTION, Math.min(s.w + dx, 1 - s.x));
      h = Math.max(MIN_FRACTION, Math.min(s.h + dy, 1 - s.y));
    }
    setRect({ x, y, w, h });
  }

  function endDrag() {
    dragRef.current = null;
  }

  async function apply() {
    setBusy(true);
    try {
      // imageOrientation "from-image" for the same reason as lib/downscale.ts:
      // canvas re-encoding drops EXIF and the server applies it, so rotation has
      // to be baked into the pixels here or portrait photos arrive sideways.
      const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
      const sx = Math.round(rect.x * bitmap.width);
      const sy = Math.round(rect.y * bitmap.height);
      const sw = Math.max(1, Math.round(rect.w * bitmap.width));
      const sh = Math.max(1, Math.round(rect.h * bitmap.height));

      const canvas = document.createElement("canvas");
      canvas.width = sw;
      canvas.height = sh;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("no 2d context");
      ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, sw, sh);
      bitmap.close();

      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.92),
      );
      if (!blob) throw new Error("encode failed");

      const base = file.name.replace(/\.[^./\\]+$/, "") || "photo";
      onCropped(new File([blob], `${base}-crop.jpg`, { type: "image/jpeg" }));
      onClose();
    } catch {
      // Cropping is optional; on failure keep the original photo and close.
      onClose();
    } finally {
      setBusy(false);
    }
  }

  const pct = (v: number) => `${v * 100}%`;

  return (
    <>
        <div
          ref={frameRef}
          onPointerMove={onMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          className="bg-muted relative max-h-[60dvh] w-full touch-none overflow-hidden rounded-xl select-none md:max-h-[62dvh]"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={src} alt="" className="max-h-[60dvh] w-full object-contain md:max-h-[62dvh]" />

          <div
            onPointerDown={(e) => beginDrag(e, "move")}
            style={{ left: pct(rect.x), top: pct(rect.y), width: pct(rect.w), height: pct(rect.h) }}
            className="absolute cursor-move"
          >
            {/* The crop window: punch the scrim out with a huge ring shadow. */}
            <div className="ring-primary absolute inset-0 shadow-[0_0_0_9999px_rgba(0,0,0,0.55)] ring-2" />
            {(["nw", "ne", "sw", "se"] as const).map((h) => (
              <button
                key={h}
                type="button"
                aria-label={`Resize ${h} corner`}
                onPointerDown={(e) => beginDrag(e, h)}
                className="bg-primary absolute size-7 rounded-full border-2 border-white/80"
                style={{
                  left: h.includes("w") ? -14 : undefined,
                  right: h.includes("e") ? -14 : undefined,
                  top: h.startsWith("n") ? -14 : undefined,
                  bottom: h.startsWith("s") ? -14 : undefined,
                }}
              />
            ))}
          </div>
        </div>

        {/* DialogFooter is flex-col-reverse below sm, so `flex-1` would apply to
            the VERTICAL axis and let both buttons shrink. The outline button was
            saved by .tap's 44px min-height; this one had only h-12, so it
            collapsed shorter than its neighbour. Full width and shrink-0 while
            stacked, sharing the row only once the footer turns horizontal. */}
        <DialogFooter className="gap-2 sm:gap-2 md:-mx-5 md:-mb-5 md:p-5">
          <Button
            variant="outline"
            onClick={onClose}
            className="h-12 w-full shrink-0 sm:flex-1 md:h-13"
          >
            Use the whole photo
          </Button>
          <Button
            onClick={apply}
            disabled={busy}
            className="h-12 w-full shrink-0 gap-2 font-semibold sm:flex-1 md:h-13"
          >
            <CheckIcon aria-hidden className="size-4" />
            {busy ? "Cropping…" : "Use this crop"}
          </Button>
        </DialogFooter>
    </>
  );
}
