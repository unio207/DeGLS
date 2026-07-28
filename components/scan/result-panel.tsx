"use client";

import { CameraIcon, LayersIcon, MessageSquareIcon, TriangleAlertIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { AnalyzeMeta, DiseaseResult, ScanInput, SeverityResult } from "@/lib/types";
import { CompareImage } from "./compare-image";
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
}

export function ResultPanel({
  view,
  onAskAssistant,
  onNewScan,
}: {
  view: ResultView;
  onAskAssistant: () => void;
  onNewScan: () => void;
}) {
  const { disease, severity, overlay, original, input, meta } = view;
  const lowConfidence = disease.confidence < LOW_CONFIDENCE;

  return (
    <section aria-live="polite" aria-labelledby="result-heading" className="animate-rise space-y-5">
      {/* The answer first, before any picture. */}
      <div className="bg-card overflow-hidden rounded-2xl border">
        <div className="border-b px-4 py-4">
          <p className="eyebrow text-muted-foreground">
            {view.recordedAt
              ? `Saved scan · ${new Date(view.recordedAt).toLocaleDateString()}`
              : "Diagnosis"}
          </p>
          <h2
            id="result-heading"
            className="font-display mt-1.5 text-[1.75rem] leading-[1.05] font-extrabold tracking-tight text-balance"
          >
            {disease.label}
          </h2>
          <p className="text-muted-foreground eyebrow tabular mt-2">
            Detection confidence {formatConfidence(disease.confidence)}
          </p>
        </div>

        <div className="p-4">
          <SeverityGauge percent={severity.percent} />
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

      {meta?.multiple_leaves && (
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

      <div className="bg-card rounded-2xl border p-4">
        {original ? (
          <CompareImage
            original={original}
            overlay={overlay}
            alt={`${disease.label} on a corn leaf`}
          />
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

      {/* The field note read back, so the saved record is verifiable at a glance. */}
      <dl className="bg-card grid grid-cols-2 gap-x-4 gap-y-3 rounded-2xl border p-4">
        <Fact label="Hybrid" value={input.corn_hybrid || "Not recorded"} />
        <Fact label="Date" value={input.date || "Not recorded"} mono />
        <Fact label="Location" value={input.location || "Not recorded"} span />
        <Fact
          label="Lesion / leaf pixels"
          value={`${severity.lesion_px.toLocaleString()} / ${severity.leaf_px.toLocaleString()}`}
          mono
          span
        />
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
          className="h-14 w-full gap-2.5 rounded-xl text-base font-semibold"
        >
          <MessageSquareIcon aria-hidden className="size-5" />
          Ask about managing this
        </Button>
        <Button
          variant="outline"
          onClick={onNewScan}
          className="tap w-full gap-2.5 rounded-xl text-[0.9375rem]"
        >
          <CameraIcon aria-hidden />
          Scan another leaf
        </Button>
      </div>
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
          "mt-1 text-[0.9375rem] leading-snug break-words",
          mono ? "tabular font-mono text-[0.8125rem]" : "font-medium",
        ].join(" ")}
      >
        {value}
      </dd>
    </div>
  );
}
