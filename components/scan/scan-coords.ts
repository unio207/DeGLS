/**
 * Coordinates for the scan map, recovered from `ScanRecord.location`.
 *
 * `ScanRecord` has no lat/lon field, so the fix is read back out of a *display
 * string*: when the locate button produces a fix, scan-app.tsx composes the
 * location as `"<place> (<lat>, <lon>)"`. When the grower types a field name by
 * hand there is no fix to recover, and that scan simply has no position.
 *
 * If a dedicated `lat`/`lon` pair is ever added to `ScanRecord`, read it here
 * instead and keep this parse only as a fallback for records written before the
 * change — there is no migration otherwise.
 */

import type { ScanRecord } from "@/lib/types";

/**
 * A trailing parenthesised decimal pair, e.g. `(41.6764, -93.6977)`.
 *
 * Anchored to the end because that is where scan-app.tsx appends it, so a place
 * name that happens to carry its own brackets cannot be mistaken for a fix.
 */
const FIX_RE = /\(\s*(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*\)\s*$/;

export interface ScanPin {
  record: ScanRecord;
  lat: number;
  lon: number;
  /** The location with the coordinates stripped — what a person reads out loud. */
  place: string;
}

export function parseFix(location: string): { lat: number; lon: number; place: string } | null {
  const match = FIX_RE.exec(location ?? "");
  if (!match) return null;

  const lat = Number(match[1]);
  const lon = Number(match[2]);
  // An out-of-range pair was never a fix, whatever else it was. Dropping it is
  // better than dropping a pin in the wrong hemisphere.
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;

  // Reverse geocoding can fail in the field, in which case the place name is
  // itself the coordinates and this leaves them as the label. That is correct:
  // it is still the most specific thing known about where the scan was taken.
  const place = location
    .slice(0, match.index)
    .trim()
    .replace(/[,\s]+$/, "");
  return { lat, lon, place };
}

/** Records without a recoverable fix are omitted — they cannot be placed. */
export function pinsFrom(records: ScanRecord[]): ScanPin[] {
  const pins: ScanPin[] = [];
  for (const record of records) {
    const fix = parseFix(record.location);
    if (fix) pins.push({ record, ...fix });
  }
  return pins;
}
