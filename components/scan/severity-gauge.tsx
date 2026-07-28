"use client";

import { useEffect, useState } from "react";

import {
  BAND_COLOR,
  BAND_LABEL,
  BAND_STOPS,
  bandOf,
  bandPosition,
  formatSeverity,
} from "./severity-scale";

/**
 * The severity readout.
 *
 * One big number in the display face, then a five-segment strip whose fill
 * carries the leaf-vein texture from the mark. The strip doubles as the axis:
 * every band is drawn at low opacity even when empty, so the reading is always
 * legible against the full range rather than as a bare number.
 */
export function SeverityGauge({ percent }: { percent: number }) {
  const band = bandOf(percent);
  const target = bandPosition(percent);
  const [fill, setFill] = useState(0);

  useEffect(() => {
    const id = requestAnimationFrame(() => setFill(target));
    return () => cancelAnimationFrame(id);
  }, [target]);

  return (
    <section aria-labelledby="severity-heading" className="space-y-3">
      <div className="flex items-baseline justify-between gap-3">
        <h3 id="severity-heading" className="eyebrow text-muted-foreground">
          Leaf area with lesions
        </h3>
        <span
          className="eyebrow rounded-full px-2 py-1 text-[color:var(--band)]"
          style={
            {
              "--band": BAND_COLOR[band],
              backgroundColor: `color-mix(in oklch, ${BAND_COLOR[band]} 16%, transparent)`,
            } as React.CSSProperties
          }
        >
          {BAND_LABEL[band]}
        </span>
      </div>

      <div className="flex items-end gap-1.5">
        <span className="tabular font-display text-[3.75rem] leading-[0.85] font-extrabold tracking-tighter">
          {formatSeverity(percent)}
        </span>
        <span className="font-display pb-1 text-2xl font-semibold opacity-70">%</span>
      </div>

      <div>
        <div
          className="border-border relative h-11 w-full overflow-hidden rounded-lg border"
          role="img"
          aria-label={`${formatSeverity(percent)} percent of leaf area, in the ${BAND_LABEL[band].toLowerCase()} display band`}
        >
          {/* Empty track. */}
          <div className="bg-secondary absolute inset-0" />

          {/* Fill, in the band's own colour and textured with the blade motif. */}
          <div
            className="absolute inset-y-0 left-0 transition-[width] duration-700 ease-out"
            style={{ width: `${fill * 100}%`, backgroundColor: BAND_COLOR[band] }}
          >
            <div className="veins-strong absolute inset-0" />
          </div>

          {/* Band boundaries, so the non-linear axis is visible rather than implied. */}
          {BAND_STOPS.slice(1, -1).map((stop, i) => (
            <div
              key={stop}
              className="absolute inset-y-0 w-px bg-[color-mix(in_oklch,var(--foreground)_28%,transparent)]"
              style={{ left: `${((i + 1) / 5) * 100}%` }}
            />
          ))}
        </div>

        <div className="text-muted-foreground relative mt-1.5 h-4">
          {BAND_STOPS.slice(1, -1).map((stop, i) => (
            <span
              key={stop}
              className="eyebrow tabular absolute -translate-x-1/2"
              style={{ left: `${((i + 1) / 5) * 100}%` }}
            >
              {stop}
            </span>
          ))}
          <span className="eyebrow absolute right-0">100%</span>
          <span className="eyebrow absolute left-0">0</span>
        </div>
      </div>

    </section>
  );
}
