"use client";

import { useEffect, useState } from "react";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { DiagnosisChat } from "@/components/chat/diagnosis-chat";
import { getConversation } from "@/lib/history";
import type { DiagnosisContext, DiagnosisUIMessage } from "@/lib/types";
import { formatSeverity } from "./severity-scale";

/**
 * The assistant gets a full-height bottom sheet rather than a box wedged under
 * the result: on a phone a conversation needs the keyboard, the scrollback and
 * the whole screen.
 */
export function ChatSheet({
  open,
  onOpenChange,
  context,
  scanId,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  context: DiagnosisContext | null;
  /** `ScanRecord.id` of the scan on screen; the thread is keyed to it. */
  scanId: string | null;
}) {
  // useChat only reads its seed messages at construction, so the saved thread
  // must be in hand before DiagnosisChat mounts. `restored` doubles as the
  // ready flag; it is keyed by scan so a stale read can't land on a new scan.
  const [restored, setRestored] = useState<{ scanId: string | null; messages: DiagnosisUIMessage[] } | null>(
    null,
  );

  useEffect(() => {
    if (!open) return;
    let alive = true;
    // Resolved through a promise even when there is no scan id, so the state
    // update never lands synchronously inside the effect body.
    const load = scanId ? getConversation(scanId) : Promise.resolve(undefined);
    load
      .then((record) => {
        if (alive) setRestored({ scanId, messages: record?.messages ?? [] });
      })
      .catch(() => {
        if (alive) setRestored({ scanId, messages: [] });
      });
    return () => {
      alive = false;
    };
  }, [open, scanId]);

  const ready = restored !== null && restored.scanId === scanId;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="bottom"
        className="safe-bottom mx-auto flex flex-col overflow-hidden rounded-t-2xl p-0 data-[side=bottom]:h-[92dvh] md:max-w-2xl md:rounded-t-3xl md:data-[side=bottom]:h-[88dvh]"
      >
        <SheetHeader className="shrink-0 border-b px-4 py-3.5 md:px-5 md:py-4">
          <SheetTitle className="font-display text-lg font-extrabold tracking-tight md:text-xl">
            Management assistant
          </SheetTitle>
          <SheetDescription className="text-[0.8125rem] md:text-sm">
            {!context
              ? "Run a scan first."
              : context.unclassified
                ? `No disease classified${context.corn_hybrid ? ` · ${context.corn_hybrid}` : ""}`
                : `${context.disease_label} · ${formatSeverity(context.severity_percent)}% of leaf${context.corn_hybrid ? ` · ${context.corn_hybrid}` : ""}`}
          </SheetDescription>
        </SheetHeader>

        <div className="min-h-0 flex-1 overflow-hidden">
          {context && ready ? (
            // Remounting per scan is deliberate: it resets useChat so scan A's
            // thread can never bleed into scan B's.
            <DiagnosisChat
              key={scanId ?? "unsaved"}
              context={context}
              scanId={scanId}
              initialMessages={restored.messages}
            />
          ) : null}
        </div>
      </SheetContent>
    </Sheet>
  );
}
