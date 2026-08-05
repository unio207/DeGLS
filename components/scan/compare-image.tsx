"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MoveHorizontalIcon } from "lucide-react";

/**
 * Original vs. lesion overlay.
 *
 * A wipe handle rather than a toggle: in the field the question is "is that
 * mark on the leaf or on the model", and answering it means seeing both at the
 * same boundary. Drag works with a thumb, arrow keys work with a keyboard, and
 * the handle is a real `slider` so it announces its position.
 */
export function CompareImage({
  original,
  overlay,
  alt,
}: {
  original: string;
  overlay: string;
  alt: string;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(55);
  const [dragging, setDragging] = useState(false);

  const setFromClientX = useCallback((clientX: number) => {
    const el = frameRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0) return;
    const next = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.min(100, Math.max(0, next)));
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const move = (e: PointerEvent) => {
      e.preventDefault();
      setFromClientX(e.clientX);
    };
    const up = () => setDragging(false);
    window.addEventListener("pointermove", move, { passive: false });
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, [dragging, setFromClientX]);

  return (
    <figure className="m-0">
      <div
        ref={frameRef}
        className="bg-secondary relative aspect-[4/5] w-full touch-none overflow-hidden rounded-xl select-none"
        onPointerDown={(e) => {
          setDragging(true);
          setFromClientX(e.clientX);
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={original}
          alt={`${alt}, original photo`}
          className="absolute inset-0 h-full w-full object-contain"
          draggable={false}
        />

        <div
          className="absolute inset-0"
          style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
          aria-hidden
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={overlay}
            alt=""
            className="absolute inset-0 h-full w-full object-contain"
            draggable={false}
          />
        </div>

        <div
          className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.45)]"
          style={{ left: `${pos}%` }}
        />

        <div
          role="slider"
          tabIndex={0}
          aria-label="Compare the lesion overlay against the original photo"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(pos)}
          aria-valuetext={`Overlay covers ${Math.round(pos)} percent of the frame`}
          onKeyDown={(e) => {
            const step = e.shiftKey ? 10 : 4;
            if (e.key === "ArrowLeft") {
              e.preventDefault();
              setPos((p) => Math.max(0, p - step));
            } else if (e.key === "ArrowRight") {
              e.preventDefault();
              setPos((p) => Math.min(100, p + step));
            } else if (e.key === "Home") {
              e.preventDefault();
              setPos(0);
            } else if (e.key === "End") {
              e.preventDefault();
              setPos(100);
            }
          }}
          className="focus-visible:ring-ring absolute top-1/2 grid size-11 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border-2 border-white bg-black/55 text-white shadow-lg backdrop-blur-sm focus-visible:ring-4 focus-visible:outline-none md:size-12"
          style={{ left: `${pos}%` }}
        >
          <MoveHorizontalIcon aria-hidden className="size-5 md:size-6" />
        </div>

        <span className="eyebrow pointer-events-none absolute bottom-2 left-2 rounded bg-black/60 px-1.5 py-1 text-white">
          Lesions
        </span>
        <span className="eyebrow pointer-events-none absolute right-2 bottom-2 rounded bg-black/60 px-1.5 py-1 text-white">
          Photo
        </span>
      </div>
      <figcaption className="text-muted-foreground mt-2 text-[0.8125rem] md:text-sm">
        Drag the handle to wipe between the model&rsquo;s lesion mask and your photo.
      </figcaption>
    </figure>
  );
}
