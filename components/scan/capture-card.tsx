"use client";

import { useRef, useState } from "react";
import { CameraIcon, CropIcon, ImageIcon, RefreshCwIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { prepareForUpload } from "@/lib/downscale";
import type { LeafPoint } from "@/lib/types";
import { CropDialog } from "./crop-dialog";

// Vercel rejects any function request body over 4.5 MB at the platform edge,
// before our code runs, with an opaque FUNCTION_PAYLOAD_TOO_LARGE. That is well
// below api/analyze.py's own 10 MB limit, so the binding constraint is the
// platform's. Measured: a 5.12 MB upload returns 413 from production.
const MAX_BYTES = 4 * 1024 * 1024;
// Target for re-encoding, with headroom under MAX_BYTES for multipart framing.
const SHRINK_TARGET = 3 * 1024 * 1024;

/**
 * Where the tap landed, in two coordinate systems.
 *
 * `point` is what the server is given: a fraction of the IMAGE. `fx`/`fy` are
 * a fraction of the PREVIEW FRAME and exist only to draw the marker. They are
 * not the same numbers whenever the photo's aspect ratio differs from 4:5,
 * because object-cover crops. Keeping the frame fraction is safe across a
 * resize precisely because the frame is always 4:5 — the visible crop of the
 * photo does not change with its width, so the fraction stays valid without
 * re-measuring on every render.
 */
export interface Tap {
  point: LeafPoint;
  fx: number;
  fy: number;
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/**
 * Map a click on the preview to a point in the photo's own pixel grid.
 *
 * The preview is `object-cover` inside a 4:5 frame, so the browser scales the
 * image by the LARGER of the two axis ratios and centres it, throwing the
 * overflow away on both sides of the long axis (object-position defaults to
 * 50% 50%). Dividing the click by the element size instead — the obvious
 * version — silently maps to the wrong pixels on any photo that isn't 4:5, and
 * a confidently wrong mask is worse than no mask at all. Worked example: a
 * 768x512 photo in a 320x400 frame scales by max(320/768, 400/512) = 0.781, so
 * it is drawn 600x400 with 140px cropped off each side; a click at x=160 (the
 * frame's middle) is 300px into a 600px-wide draw, i.e. 0.5 of the image, but
 * a click at x=0 is image x = 140/600 = 0.233, not 0.
 */
function tapToImage(e: React.MouseEvent, img: HTMLImageElement): Tap | null {
  const box = img.getBoundingClientRect();
  const { naturalWidth: nw, naturalHeight: nh } = img;
  if (!nw || !nh || !box.width || !box.height) return null;

  const scale = Math.max(box.width / nw, box.height / nh);
  const drawnW = nw * scale;
  const drawnH = nh * scale;
  const offsetX = (box.width - drawnW) / 2;
  const offsetY = (box.height - drawnH) / 2;

  // detail === 0 means the button was fired from the keyboard, where clientX/Y
  // are meaningless. The frame centre is the honest answer there, and on a
  // centred cover crop the frame centre is also the image centre.
  const keyboard = e.detail === 0;
  const elX = keyboard ? box.width / 2 : e.clientX - box.left;
  const elY = keyboard ? box.height / 2 : e.clientY - box.top;

  return {
    point: {
      x: clamp01((elX - offsetX) / drawnW),
      y: clamp01((elY - offsetY) / drawnH),
    },
    fx: clamp01(elX / box.width),
    fy: clamp01(elY / box.height),
  };
}

/**
 * The capture surface.
 *
 * Camera first: the person using this is standing next to the leaf, so the
 * rear camera is the primary action and the library picker is the fallback.
 * The frame keeps a fixed 4:5 aspect whether it is empty or holding a preview,
 * so choosing a photo never shifts the form below it.
 *
 * Once a photo is in, the frame is also the tap target for marking the leaf.
 * That tap is what tells the server which blade to segment; without it the
 * server falls back to the frame centre, which was wrong on 2 of 6 real field
 * photos, one of them by 17 severity points.
 */
export function CaptureCard({
  previewUrl,
  file,
  onSelect,
  onReject,
  tap,
  onTap,
  disabled,
}: {
  previewUrl: string | null;
  /** The upload copy, used as the crop source when the original is gone. */
  file: File | null;
  onSelect: (file: File) => void;
  onReject: (message: string) => void;
  tap: Tap | null;
  /** Null whenever the photo changes: the old point refers to different pixels. */
  onTap: (tap: Tap | null) => void;
  disabled?: boolean;
}) {
  const cameraRef = useRef<HTMLInputElement>(null);
  const libraryRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [dragging, setDragging] = useState(false);
  // The photo as picked, kept so the crop dialog works on the best copy we
  // have rather than on the already-downscaled upload.
  const [picked, setPicked] = useState<File | null>(null);
  const [cropOpen, setCropOpen] = useState(false);
  // `picked` is lost whenever this card unmounts — which now happens when a
  // failed scan sends the user back to re-tap — so fall back to the upload
  // copy. It is capped at 2048px by prepareForUpload rather than being the
  // camera original, which is still far more resolution than the models use.
  const cropSource = picked ?? file;

  async function accept(file: File | undefined | null) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      onReject("That file isn't an image. Pick a JPEG, PNG or HEIC photo.");
      return;
    }
    setPicked(file);
    // Every new photo arrives here, including the cropped copy from CropDialog.
    // A crop re-frames the pixels, so a point taken before it now names a
    // different part of the leaf — clearing on any change is the only correct
    // rule, and it puts the "Tap the leaf" prompt back on screen.
    onTap(null);

    // Every photo goes through this, not just oversized ones: the server runs
    // out of memory on full-resolution frames regardless of file size, because
    // what costs memory is pixel count. prepareForUpload returns the original
    // untouched when it is already small enough in both dimensions and bytes.
    const prepared = await prepareForUpload(file, SHRINK_TARGET);
    if (prepared) {
      onSelect(prepared);
      return;
    }
    if (file.size > MAX_BYTES) {
      onReject("That photo is too large and couldn't be resized. Take a new one at a lower resolution.");
      return;
    }
    onSelect(file);
  }

  return (
    <section aria-labelledby="capture-heading" className="space-y-3">
      <h2 id="capture-heading" className="sr-only">
        Leaf photo
      </h2>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void accept(e.dataTransfer.files?.[0]);
        }}
        className={[
          "relative aspect-[4/5] w-full overflow-hidden rounded-2xl border-2 border-dashed transition-colors",
          dragging ? "border-primary bg-primary/10" : "border-input bg-muted",
        ].join(" ")}
      >
        {previewUrl ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={imgRef}
              src={previewUrl}
              alt="The leaf photo you selected"
              className="absolute inset-0 h-full w-full object-cover"
            />

            {/* The whole frame is the target — a thumb needs no aiming to hit
                it, and the marker then shows what was actually understood. */}
            <button
              type="button"
              disabled={disabled}
              aria-label={
                tap ? "Move the marker to the leaf you want measured" : "Tap the leaf you want measured"
              }
              onClick={(e) => {
                if (!imgRef.current) return;
                const next = tapToImage(e, imgRef.current);
                if (next) onTap(next);
              }}
              className="absolute inset-0 cursor-crosshair"
            />

            {tap ? (
              <span
                aria-hidden
                style={{ left: `${tap.fx * 100}%`, top: `${tap.fy * 100}%` }}
                className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2"
              >
                <span className="border-primary block size-9 rounded-full border-[3px] bg-white/25 shadow-[0_0_0_2px_rgba(255,255,255,0.9)] md:size-10" />
                <span className="bg-primary absolute top-1/2 left-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-white/90" />
              </span>
            ) : (
              // Dimming the photo until it is tapped is the instruction: the
              // frame reads as unfinished, and the prompt sits where the thumb
              // already is rather than as a caption someone has to read first.
              <span className="pointer-events-none absolute inset-0 grid place-items-center bg-black/35">
                <span className="font-display rounded-full bg-white px-4 py-2 text-[0.9375rem] font-bold text-black shadow-lg md:px-5 md:py-2.5 md:text-base">
                  Tap the leaf
                </span>
              </span>
            )}

            <div className="absolute right-3 bottom-3 flex gap-2 md:right-4 md:bottom-4">
              <Button
                variant="outline"
                onClick={() => setCropOpen(true)}
                disabled={disabled || !cropSource}
                className="tap gap-2 px-3 shadow-md"
              >
                <CropIcon aria-hidden />
                Crop
              </Button>
              <Button
                variant="outline"
                onClick={() => cameraRef.current?.click()}
                disabled={disabled}
                className="tap gap-2 px-3 shadow-md"
              >
                <RefreshCwIcon aria-hidden />
                Retake
              </Button>
            </div>
          </>
        ) : (
          <div className="absolute inset-0 grid place-items-center px-6 text-center">
            <div>
              <div className="veins bg-secondary text-primary mx-auto grid size-16 place-items-center rounded-2xl md:size-20">
                <CameraIcon aria-hidden className="size-8 md:size-10" />
              </div>
              <p className="font-display mt-4 text-lg leading-tight font-bold md:mt-5 md:text-2xl">
                Photograph one leaf
              </p>
              <p className="text-muted-foreground mx-auto mt-1.5 max-w-[26ch] text-sm leading-snug md:mt-2.5 md:text-base">
                Fill the frame with the blade. Even light, no hard shadow, plain background if you
                can manage one.
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-2">
        <Button
          onClick={() => cameraRef.current?.click()}
          disabled={disabled}
          className="h-14 w-full gap-2.5 rounded-xl text-base font-semibold md:h-15 md:text-lg"
        >
          <CameraIcon aria-hidden className="size-5 md:size-6" />
          {previewUrl ? "Take another photo" : "Take a photo"}
        </Button>
        <Button
          variant="outline"
          onClick={() => libraryRef.current?.click()}
          disabled={disabled}
          className="h-12 w-full gap-2.5 rounded-xl text-[0.9375rem] md:h-13 md:text-base"
        >
          <ImageIcon aria-hidden />
          Choose from library
        </Button>
      </div>

      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        tabIndex={-1}
        aria-hidden
        onChange={(e) => {
          void accept(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <input
        ref={libraryRef}
        type="file"
        accept="image/*"
        className="sr-only"
        tabIndex={-1}
        aria-hidden
        onChange={(e) => {
          void accept(e.target.files?.[0]);
          e.target.value = "";
        }}
      />

      <CropDialog
        open={cropOpen}
        onOpenChange={setCropOpen}
        file={cropSource}
        onCropped={(cropped) => void accept(cropped)}
      />
    </section>
  );
}
