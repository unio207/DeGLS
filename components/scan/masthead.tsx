"use client";

import Image from "next/image";
import { HistoryIcon, MapPinnedIcon } from "lucide-react";

import { Button } from "@/components/ui/button";

/**
 * Slim sticky masthead. The blade texture behind it is the one piece of
 * ornament in the app — parallel corn-leaf veins lifted from the mark.
 */
export function Masthead({
  onOpenHistory,
  onOpenMap,
  historyCount,
}: {
  onOpenHistory: () => void;
  onOpenMap: () => void;
  historyCount: number;
}) {
  return (
    <header className="bg-background/85 safe-top safe-x sticky top-0 z-30 border-b backdrop-blur-md">
      <div className="veins pointer-events-none absolute inset-0 opacity-30" aria-hidden />
      {/* Same cap as <main> so the mark and the content share a left edge. */}
      <div className="relative mx-auto flex w-full max-w-2xl items-center gap-2.5 px-4 py-2.5 md:max-w-5xl md:gap-3 md:px-6 md:py-3">
        <Image
          src="/icons/icon-192.png"
          alt=""
          width={44}
          height={44}
          className="size-9 rounded-lg md:size-11"
          priority
        />
        <div className="min-w-0 flex-1">
          <p className="font-display text-[1.0625rem] leading-none font-extrabold tracking-tight md:text-[1.375rem]">
            DeGLS
          </p>
          <p className="eyebrow text-muted-foreground mt-1 truncate">Corn leaf disease field scan</p>
        </div>
        {/* Icon only — the count belongs to history, and a second number here
            would read as a different tally of the same scans. */}
        <Button
          variant="outline"
          onClick={onOpenMap}
          className="size-11 md:size-12"
          aria-label="Map of past scans"
        >
          <MapPinnedIcon aria-hidden />
        </Button>
        <Button
          variant="outline"
          onClick={onOpenHistory}
          className="tap gap-2 px-3 md:px-4"
          aria-label={
            historyCount > 0 ? `Past scans, ${historyCount} saved` : "Past scans, none saved yet"
          }
        >
          <HistoryIcon aria-hidden />
          <span className="eyebrow tabular">{historyCount > 0 ? historyCount : "0"}</span>
        </Button>
      </div>
    </header>
  );
}
