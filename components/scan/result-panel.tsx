"use client";

import { CameraIcon, LayersIcon, MessageSquareIcon, TriangleAlertIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { AnalyzeMeta, DiseaseResult, ScanInput, SeverityResult } from "@/lib/types";
import { CompareImage } from "./compare-image";
import { DEMO_HEALTHY_TEXT } from "./demo-healthy";
import { SeverityGauge } from "./severity-gauge";
import { LOW_CONFIDENCE, formatConfidence } from "./severity-scale";

export interface ResultView {
  disease: DiseaseResult;
  severity: SeverityResult;
  overlay: string;
  /** The untouched photo, when we still have it. History records keep only the overlay. */
  original: string | null;
  input: ScanInput;
  meta: AnalyzeMeta | null;
  /** Epoch ms, for records read back out of history. */
  recordedAt?: number;
  /** Temporary demo override — see demo-healthy.ts. Remove after the demo. */
  presentAsHealthy?: boolean;
}

export function ResultPanel({
  view,
  onAskAssistant,
  hasConversation = false,
  onNewScan,
}: {
  view: ResultView;
  onAskAssistant: () => void;
  /** A saved thread exists for this scan, so the entry point resumes it. */
  hasConversation?: boolean;
  onNewScan: () => void;
}) {
  const { disease, severity, overlay, original, input, meta } = view;

  // Temporary demo override (see demo-healthy.ts). Everything that would name a
  // disease or put a number on the severity is suppressed together — a healthy
  // headline over a confidence score and a lesion pixel count would read as a
  // bug. Remove this line and its uses below after the demo.
  const healthy = view.presentAsHealthy === true;

  const lowConfidence = !healthy && disease.confidence < LOW_CONFIDENCE;
  const imageAlt = healthy ? "The scanned corn leaf" : `${disease.label} on a corn leaf`;

  // A reopened history record under the demo override keeps only the overlay,
  // which that branch declines to show — so there is nothing for the right-hand
  // column to hold, and the tablet layout stays a single centred column rather
  // than a half-width panel next to a void.
  const hasPicture = healthy ? original !== null : true;

  return (
    /*
      Phone: a flex column in exactly the order it has always been — answer,
      picture, record. From 768px the same three blocks become an explicitly
      placed two-column grid, answer over record on the left and the picture
      spanning both rows on the right. Explicit placement rather than a DOM
      reorder is what keeps the phone order and the reading order identical.
    */
    <section
      aria-live="polite"
      aria-labelledby="result-heading"
      className={[
        "animate-rise flex flex-col gap-5",
        hasPicture
          ? "md:grid md:grid-cols-2 md:items-start md:gap-x-6 md:gap-y-5"
          : "md:mx-auto md:max-w-2xl",
      ].join(" ")}
    >
      {/* The answer first, before any picture. */}
      <div className="space-y-5 md:col-start-1 md:row-start-1">
      <div className="bg-card overflow-hidden rounded-2xl border">
        <div className="border-b px-4 py-4 md:px-5 md:py-5">
          <p className="eyebrow text-muted-foreground">
            {view.recordedAt
              ? `Saved scan · ${new Date(view.recordedAt).toLocaleDateString()}`
              : "Diagnosis"}
          </p>
          <h2
            id="result-heading"
            className="font-display mt-1.5 text-[1.75rem] leading-[1.05] font-extrabold tracking-tight text-balance md:text-[2.125rem]"
          >
            {healthy ? "Healthy" : disease.label}
          </h2>
          {!healthy && (
            <p className="text-muted-foreground eyebrow tabular mt-2">
              Detection confidence {formatConfidence(disease.confidence)}
            </p>
          )}
        </div>

        <div className="p-4 md:p-5">
          {healthy ? <HealthyReadout /> : <SeverityGauge percent={severity.percent} />}
        </div>
      </div>

      {lowConfidence && (
        <div
          className="border-destructive/40 bg-destructive/10 text-foreground flex gap-3 rounded-xl border-2 p-3.5"
          role="note"
        >
          <TriangleAlertIcon aria-hidden className="text-destructive mt-0.5 size-5 shrink-0" />
          <div>
            <p className="font-display text-[0.9375rem] leading-tight font-bold">
              Low confidence — {formatConfidence(disease.confidence)}
            </p>
            <p className="mt-1 text-[0.875rem] leading-snug">
              The model is not sure this is {disease.label}. Shoot the same leaf again, closer and
              squarer, before you record it — and check a second leaf from the same plant.
            </p>
          </div>
        </div>
      )}

      {!healthy && meta?.multiple_leaves && (
        <div
          className="bg-secondary text-secondary-foreground flex gap-3 rounded-xl p-3.5"
          role="note"
        >
          <LayersIcon aria-hidden className="mt-0.5 size-5 shrink-0" />
          <div>
            <p className="font-display text-[0.9375rem] leading-tight font-bold">
              {meta.instances_detected} leaves in this photo
            </p>
            <p className="mt-1 text-[0.875rem] leading-snug">
              The severity above covers only the highest-confidence leaf, not the whole frame.
              Photograph the others separately for a reading on each.
            </p>
          </div>
        </div>
      )}
      </div>

      {/*
        Under the demo override the server overlay is never rendered: the red
        lesion mask is baked into that PNG, so a panel saying nothing was
        classified would be showing lesions anyway. The original photo takes its
        place, plain — a wipe between two identical frames has nothing to reveal,
        so the comparer, its handle and its labels all go. Reopened history
        records keep only the overlay, so there is no photo to fall back to and
        the picture is dropped entirely rather than shown in red. Remove this
        branch after the demo; the `else` below is the real behaviour.
      */}
      {healthy ? (
        original && (
          <div className="bg-card rounded-2xl border p-4 md:col-start-2 md:row-start-1 md:row-span-2">
            <div className="bg-secondary aspect-[4/5] w-full overflow-hidden rounded-xl">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={original} alt={imageAlt} className="h-full w-full object-contain" />
            </div>
          </div>
        )
      ) : (
        <div className="bg-card rounded-2xl border p-4 md:col-start-2 md:row-start-1 md:row-span-2">
          {original ? (
            <CompareImage original={original} overlay={overlay} alt={imageAlt} />
          ) : (
            <figure className="m-0">
              <div className="bg-secondary aspect-[4/5] w-full overflow-hidden rounded-xl">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={overlay}
                  alt={`Lesion overlay for ${disease.label}`}
                  className="h-full w-full object-contain"
                />
              </div>
              <figcaption className="text-muted-foreground mt-2 text-[0.8125rem]">
                Saved scans keep the overlay only, so there is no original to compare against.
              </figcaption>
            </figure>
          )}
        </div>
      )}

      <div className="space-y-5 md:col-start-1 md:row-start-2">
      {/* The field note read back, so the saved record is verifiable at a glance. */}
      <dl className="bg-card grid grid-cols-2 gap-x-4 gap-y-3 rounded-2xl border p-4 md:p-5">
        <Fact label="Hybrid" value={input.corn_hybrid || "Not recorded"} />
        <Fact label="Date" value={input.date || "Not recorded"} mono />
        <Fact label="Location" value={input.location || "Not recorded"} span />
        {!healthy && (
          <Fact
            label="Lesion / leaf pixels"
            value={`${severity.lesion_px.toLocaleString()} / ${severity.leaf_px.toLocaleString()}`}
            mono
            span
          />
        )}
        {meta && (
          <>
            <Fact label="Processing" value={`${(meta.processing_ms / 1000).toFixed(1)}s`} mono />
            <Fact label="Leaves found" value={String(meta.instances_detected)} mono />
          </>
        )}
      </dl>

      <div className="grid gap-2">
        <Button
          onClick={onAskAssistant}
          className="h-14 w-full gap-2.5 rounded-xl text-base font-semibold md:h-15 md:text-lg"
        >
          <MessageSquareIcon aria-hidden className="size-5 md:size-6" />
          {hasConversation ? "Continue chatting" : "Ask about managing this"}
        </Button>
        <Button
          variant="outline"
          onClick={onNewScan}
          className="h-12 w-full gap-2.5 rounded-xl text-[0.9375rem] md:h-13 md:text-base"
        >
          <CameraIcon aria-hidden />
          Scan another leaf
        </Button>
      </div>
      </div>
    </section>
  );
}

/**
 * Stands in for the gauge under the demo override (see demo-healthy.ts). Keeps
 * the gauge's header row so the card's rhythm does not change, with the message
 * sitting where the big number would. Remove after the demo.
 */
function HealthyReadout() {
  return (
    <section aria-labelledby="severity-heading" className="space-y-3">
      <h3 id="severity-heading" className="eyebrow text-muted-foreground">
        Leaf area with lesions
      </h3>
      <p className="font-display text-[1.25rem] leading-tight font-bold text-balance md:text-[1.5rem]">
        {DEMO_HEALTHY_TEXT}
      </p>
    </section>
  );
}

function Fact({
  label,
  value,
  mono,
  span,
}: {
  label: string;
  value: string;
  mono?: boolean;
  span?: boolean;
}) {
  return (
    <div className={span ? "col-span-2 min-w-0" : "min-w-0"}>
      <dt className="eyebrow text-muted-foreground">{label}</dt>
      <dd
        className={[
          "mt-1 text-[0.9375rem] leading-snug break-words md:text-base",
          mono ? "tabular font-mono text-[0.8125rem]" : "font-medium",
        ].join(" ")}
      >
        {value}
      </dd>
    </div>
  );
}
