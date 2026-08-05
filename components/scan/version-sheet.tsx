"use client";

import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

/**
 * Version history.
 *
 * Hand-maintained rather than generated from git: the useful record is what
 * changed about the *reading*, which a commit list does not say. Add an entry
 * whenever behaviour that affects a number or a decision changes; skip pure
 * refactors and copy tweaks.
 *
 * Newest first. Keep `notes` to one line a person in a field would care about.
 */
export interface VersionEntry {
  version: string;
  date: string;
  notes: string[];
}

export const VERSIONS: VersionEntry[] = [
  {
    version: "0.7.0",
    date: "2026-08-05",
    notes: [
      "New map of past scans, with pins coloured by severity. Scans taken with the locate button appear on it.",
      "Location now names your town instead of the civil township — it was reading \"Township of Webster\" for Johnston.",
      "Assistant answers are much shorter, and each leaf keeps its own conversation so you can pick up where you left off.",
      "Laid out for iPad, and fixed content sitting flush against the screen edge on every device.",
    ],
  },
  {
    version: "0.6.0",
    date: "2026-08-05",
    notes: [
      "Added a crop tool: trim the photo to the leaf before scanning, so background weeds and soil are not measured.",
    ],
  },
  {
    version: "0.5.0",
    date: "2026-08-04",
    notes: [
      "Fixed full-resolution photos failing outright: the server ran out of memory on large frames, and anything over 4.5 MB was rejected before it arrived.",
      "Photos are now capped at 2048px on the long edge, on the phone and again on the server.",
    ],
  },
  {
    version: "0.4.0",
    date: "2026-08-04",
    notes: [
      "Highlight only the largest contiguous region of the detected leaf, so scattered fragments on weeds and soil are no longer measured.",
      "Raised the implausible-severity ceiling to 45% after finding it was suppressing a genuinely diseased leaf at 39%.",
    ],
  },
  {
    version: "0.3.0",
    date: "2026-08-04",
    notes: [
      "Oversized photos are resized in the browser instead of being refused.",
      "Client and server size limits now agree at 10 MB.",
    ],
  },
  {
    version: "0.2.0",
    date: "2026-08-04",
    notes: [
      "Readings above the plausibility ceiling are withheld: uniform yellowing was being reported as severe disease.",
    ],
  },
  {
    version: "0.1.0",
    date: "2026-07-28",
    notes: ["Rebuilt on Next.js and ONNX. Leaf-plausibility guard for non-leaf photos."],
  },
];

/**
 * Vercel exposes the deployed commit to the browser as this variable when
 * system environment variables are enabled, which they are by default. Locally
 * it is undefined, hence the fallback — a dev build should not claim a SHA.
 */
const COMMIT = process.env.NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA?.slice(0, 7) ?? null;

export const CURRENT_VERSION = VERSIONS[0].version;

export function VersionSheet({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="bottom"
        className="safe-bottom mx-auto flex flex-col overflow-hidden rounded-t-2xl p-0 data-[side=bottom]:h-[70dvh] md:max-w-xl md:rounded-t-3xl md:data-[side=bottom]:h-[62dvh]"
      >
        <SheetHeader className="shrink-0 border-b px-4 py-3.5 md:px-5 md:py-4">
          <SheetTitle className="font-display text-lg font-extrabold tracking-tight md:text-xl">
            Version history
          </SheetTitle>
          <SheetDescription className="text-[0.8125rem] md:text-sm">
            What changed about the reading, newest first.
            {COMMIT ? ` Running build ${COMMIT}.` : " Running a local build."}
          </SheetDescription>
        </SheetHeader>

        <ScrollArea className="min-h-0 flex-1">
          <ol className="space-y-4 p-4">
            {VERSIONS.map((entry, i) => (
              <li key={entry.version} className="border-b pb-4 last:border-b-0 last:pb-0">
                <div className="flex items-baseline gap-2">
                  <span className="font-display text-[0.9375rem] font-bold tabular-nums">
                    v{entry.version}
                  </span>
                  <span className="eyebrow text-muted-foreground tabular">{entry.date}</span>
                  {i === 0 && (
                    <span className="bg-secondary text-secondary-foreground eyebrow rounded px-1.5 py-0.5">
                      current
                    </span>
                  )}
                </div>
                <ul className="mt-1.5 space-y-1.5">
                  {entry.notes.map((note) => (
                    <li key={note} className="flex gap-2 text-[0.875rem] leading-snug">
                      <span
                        className="bg-muted-foreground/40 mt-[0.4rem] size-1 shrink-0 rounded-full"
                        aria-hidden
                      />
                      <span>{note}</span>
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ol>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
