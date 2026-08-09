"use client";

import { useEffect, useState } from "react";
import { CheckIcon, LoaderCircleIcon } from "lucide-react";

/**
 * Wait feedback.
 *
 * Only the upload has a real denominator, so only the upload gets a bar. Once
 * the bytes are gone the server does detection and segmentation without
 * reporting back, so those two rows light up together under one indeterminate
 * sweep rather than pretending to advance on a timer. The elapsed counter is
 * the honest substitute for a percentage.
 */
export function ProgressStages({
  uploadFraction,
  uploaded,
}: {
  uploadFraction: number;
  uploaded: boolean;
}) {
  const [elapsed, setElapsed] = useState(0);

  // The clock starts when this panel mounts, which is the moment the request
  // leaves — no need to thread a timestamp through component state.
  useEffect(() => {
    const startedAt = Date.now();
    const id = setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 100);
    return () => clearInterval(id);
  }, []);

  return (
    <section
      aria-live="polite"
      aria-busy="true"
      className="bg-card animate-rise rounded-2xl border p-4 md:p-5"
    >
      <div className="flex items-baseline justify-between">
        <h2 className="eyebrow text-muted-foreground">Working</h2>
        <span className="eyebrow tabular text-muted-foreground">{elapsed.toFixed(1)}s</span>
      </div>

      <ol className="mt-4 space-y-3.5">
        <Stage
          state={uploaded ? "done" : "active"}
          label="Uploading the photo"
          detail={uploaded ? "Sent" : `${Math.round(uploadFraction * 100)}% sent`}
        >
          {!uploaded && (
            <div className="bg-secondary mt-2 h-1.5 w-full overflow-hidden rounded-full">
              <div
                className="bg-primary h-full rounded-full transition-[width] duration-200 ease-out"
                style={{ width: `${Math.max(2, uploadFraction * 100)}%` }}
              />
            </div>
          )}
        </Stage>

        <Stage
          state={uploaded ? "active" : "waiting"}
          label="Finding the leaf"
          detail={uploaded ? "Running" : "Waiting for the upload"}
        />

        <Stage
          state={uploaded ? "active" : "waiting"}
          label="Measuring lesion area"
          detail={uploaded ? "Running" : "Waiting for the upload"}
        />
      </ol>

      {uploaded && (
        <>
          <div className="bg-secondary relative mt-4 h-1.5 w-full overflow-hidden rounded-full">
            <div className="bg-primary animate-sweep absolute inset-y-0 w-1/3 rounded-full" />
          </div>
          <p className="text-muted-foreground mt-2 text-[0.8125rem] leading-snug">
            Both results come back together. Usually 2–5 seconds.
          </p>
        </>
      )}
    </section>
  );
}

function Stage({
  state,
  label,
  detail,
  children,
}: {
  state: "waiting" | "active" | "done";
  label: string;
  detail: string;
  children?: React.ReactNode;
}) {
  return (
    <li className="flex gap-3">
      <span
        aria-hidden
        className={[
          "mt-0.5 grid size-6 shrink-0 place-items-center rounded-full border",
          state === "done"
            ? "bg-primary border-primary text-primary-foreground"
            : state === "active"
              ? "border-primary text-primary"
              : "border-border text-muted-foreground",
        ].join(" ")}
      >
        {state === "done" ? (
          <CheckIcon className="size-3.5" />
        ) : state === "active" ? (
          <LoaderCircleIcon className="size-3.5 animate-spin" />
        ) : (
          <span className="bg-current size-1.5 rounded-full opacity-50" />
        )}
      </span>
      <div className="min-w-0 flex-1">
        <p
          className={[
            "text-[0.9375rem] leading-tight font-medium md:text-base",
            state === "waiting" ? "text-muted-foreground" : "text-foreground",
          ].join(" ")}
        >
          {label}
        </p>
        <p className="text-muted-foreground mt-0.5 text-[0.8125rem] leading-tight">{detail}</p>
        {children}
      </div>
    </li>
  );
}
