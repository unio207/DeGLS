"use client";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { DiagnosisChat } from "@/components/chat/diagnosis-chat";
import type { DiagnosisContext } from "@/lib/types";
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
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  context: DiagnosisContext | null;
}) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="bottom"
        className="safe-bottom flex flex-col overflow-hidden rounded-t-2xl p-0 data-[side=bottom]:h-[92dvh]"
      >
        <SheetHeader className="shrink-0 border-b px-4 py-3.5">
          <SheetTitle className="font-display text-lg font-extrabold tracking-tight">
            Management assistant
          </SheetTitle>
          <SheetDescription className="text-[0.8125rem]">
            {!context
              ? "Run a scan first."
              : context.unclassified
                ? `No disease was classified on this scan${context.corn_hybrid ? ` of ${context.corn_hybrid}` : ""}.`
                : `Answering about ${context.disease_label} at ${formatSeverity(context.severity_percent)}% leaf area${context.corn_hybrid ? ` on ${context.corn_hybrid}` : ""}.`}
          </SheetDescription>
        </SheetHeader>

        <div className="min-h-0 flex-1 overflow-hidden">
          {context ? <DiagnosisChat context={context} /> : null}
        </div>
      </SheetContent>
    </Sheet>
  );
}
