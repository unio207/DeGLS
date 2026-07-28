"use client";

import { CrosshairIcon, LoaderCircleIcon, XIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { coordText, type Fix, type PlaceStatus } from "./use-place";

/**
 * The three things that turn a photo into a record: which hybrid, which field,
 * which day. Laid out as a field-book entry — mono labels, generous rows — and
 * every control is at least 44px tall for a gloved thumb.
 */
export function FieldNote({
  hybrid,
  onHybridChange,
  location,
  onLocationChange,
  fix,
  onClearFix,
  date,
  onDateChange,
  onLocate,
  placeStatus,
  placeMessage,
  disabled,
}: {
  hybrid: string;
  onHybridChange: (v: string) => void;
  location: string;
  onLocationChange: (v: string) => void;
  fix: Fix | null;
  onClearFix: () => void;
  date: string;
  onDateChange: (v: string) => void;
  onLocate: () => void;
  placeStatus: PlaceStatus;
  placeMessage: string | null;
  disabled?: boolean;
}) {
  const locating = placeStatus === "locating" || placeStatus === "naming";

  return (
    <section aria-labelledby="fieldnote-heading" className="space-y-4">
      <div className="flex items-center gap-3">
        <h2 id="fieldnote-heading" className="eyebrow text-muted-foreground">
          Field note
        </h2>
        <span className="bg-border h-px flex-1" aria-hidden />
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="corn-hybrid" className="eyebrow text-muted-foreground">
          Corn hybrid
        </Label>
        <Input
          id="corn-hybrid"
          name="corn_hybrid"
          value={hybrid}
          onChange={(e) => onHybridChange(e.target.value)}
          placeholder="e.g. P1197AM"
          autoComplete="off"
          autoCapitalize="characters"
          spellCheck={false}
          disabled={disabled}
          className="h-12 rounded-xl text-base"
        />
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="location" className="eyebrow text-muted-foreground">
          Location
        </Label>
        <div className="flex gap-2">
          <Input
            id="location"
            name="location"
            value={location}
            onChange={(e) => {
              onLocationChange(e.target.value);
              if (fix) onClearFix();
            }}
            placeholder="Field or town"
            autoComplete="off"
            disabled={disabled}
            className="h-12 flex-1 rounded-xl text-base"
          />
          <Button
            variant="outline"
            onClick={onLocate}
            disabled={disabled || locating}
            aria-label="Use my current location"
            className="size-12 shrink-0 rounded-xl"
          >
            {locating ? (
              <LoaderCircleIcon aria-hidden className="size-5 animate-spin" />
            ) : (
              <CrosshairIcon aria-hidden className="size-5" />
            )}
          </Button>
        </div>

        <p className="min-h-5 text-[0.8125rem] leading-snug" aria-live="polite">
          {locating ? (
            <span className="text-muted-foreground">
              {placeStatus === "locating" ? "Getting a fix…" : "Looking up the place name…"}
            </span>
          ) : fix ? (
            <span className="text-muted-foreground inline-flex items-center gap-1.5">
              <span className="font-mono tabular text-[0.75rem]">
                {coordText(fix.lat, fix.lon)}
              </span>
              <button
                type="button"
                onClick={onClearFix}
                className="hover:text-foreground focus-visible:ring-ring inline-flex items-center gap-0.5 rounded underline underline-offset-2 focus-visible:ring-2 focus-visible:outline-none"
              >
                <XIcon aria-hidden className="size-3" />
                drop coordinates
              </button>
            </span>
          ) : placeMessage ? (
            <span className="text-destructive">{placeMessage}</span>
          ) : null}
        </p>
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="date" className="eyebrow text-muted-foreground">
          Date
        </Label>
        <Input
          id="date"
          name="date"
          type="date"
          value={date}
          onChange={(e) => onDateChange(e.target.value)}
          disabled={disabled}
          className="tabular h-12 w-full rounded-xl text-base"
        />
      </div>
    </section>
  );
}
