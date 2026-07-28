"use client";

import Image from "next/image";
import { HistoryIcon } from "lucide-react";

import { Button } from "@/components/ui/button";

/**
 * Slim sticky masthead. The blade texture behind it is the one piece of
 * ornament in the app — parallel corn-leaf veins lifted from the mark.
 */
export function Masthead({
  onOpenHistory,
  historyCount,
}: {
  onOpenHistory: () => void;
  historyCount: number;
}) {
  return (
    <header className="bg-background/85 safe-top safe-x sticky top-0 z-30 border-b backdrop-blur-md">
      <div className="veins pointer-events-none absolute inset-0 opacity-30" aria-hidden />
      <div className="relative mx-auto flex w-full max-w-2xl items-center gap-2.5 px-4 py-2.5">
        <Image
          src="/icons/icon-192.png"
          alt=""
          width={36}
          height={36}
          className="size-9 rounded-lg"
          priority
        />
        <div className="min-w-0 flex-1">
          <p className="font-display text-[1.0625rem] leading-none font-extrabold tracking-tight">
            DeGLS
          </p>
          <p className="eyebrow text-muted-foreground mt-1 truncate">Corn leaf disease field scan</p>
        </div>
        <Button
          variant="outline"
          onClick={onOpenHistory}
          className="tap gap-2 px-3"
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
