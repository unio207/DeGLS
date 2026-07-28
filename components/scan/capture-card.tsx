"use client";

import { useRef, useState } from "react";
import { CameraIcon, ImageIcon, RefreshCwIcon } from "lucide-react";

import { Button } from "@/components/ui/button";

const MAX_BYTES = 12 * 1024 * 1024;

/**
 * The capture surface.
 *
 * Camera first: the person using this is standing next to the leaf, so the
 * rear camera is the primary action and the library picker is the fallback.
 * The frame keeps a fixed 4:5 aspect whether it is empty or holding a preview,
 * so choosing a photo never shifts the form below it.
 */
export function CaptureCard({
  previewUrl,
  onSelect,
  onReject,
  disabled,
}: {
  previewUrl: string | null;
  onSelect: (file: File) => void;
  onReject: (message: string) => void;
  disabled?: boolean;
}) {
  const cameraRef = useRef<HTMLInputElement>(null);
  const libraryRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function accept(file: File | undefined | null) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      onReject("That file isn't an image. Pick a JPEG, PNG or HEIC photo.");
      return;
    }
    if (file.size > MAX_BYTES) {
      onReject("That photo is over 12 MB. Take a new one at a lower resolution.");
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
          accept(e.dataTransfer.files?.[0]);
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
              src={previewUrl}
              alt="The leaf photo you selected"
              className="absolute inset-0 h-full w-full object-cover"
            />
            <Button
              variant="outline"
              onClick={() => cameraRef.current?.click()}
              disabled={disabled}
              className="tap absolute right-3 bottom-3 gap-2 px-3 shadow-md"
            >
              <RefreshCwIcon aria-hidden />
              Retake
            </Button>
          </>
        ) : (
          <div className="absolute inset-0 grid place-items-center px-6 text-center">
            <div>
              <div className="veins bg-secondary text-primary mx-auto grid size-16 place-items-center rounded-2xl">
                <CameraIcon aria-hidden className="size-8" />
              </div>
              <p className="font-display mt-4 text-lg leading-tight font-bold">
                Photograph one leaf
              </p>
              <p className="text-muted-foreground mx-auto mt-1.5 max-w-[26ch] text-sm leading-snug">
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
          className="h-14 w-full gap-2.5 rounded-xl text-base font-semibold"
        >
          <CameraIcon aria-hidden className="size-5" />
          {previewUrl ? "Take another photo" : "Take a photo"}
        </Button>
        <Button
          variant="outline"
          onClick={() => libraryRef.current?.click()}
          disabled={disabled}
          className="tap w-full gap-2.5 rounded-xl text-[0.9375rem]"
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
          accept(e.target.files?.[0]);
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
          accept(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
    </section>
  );
}
