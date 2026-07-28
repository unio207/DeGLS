"use client";

import { useEffect, useState } from "react";
import { HistoryIcon, Trash2Icon } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import type { ScanRecord } from "@/lib/types";
import { deleteScan, listScans } from "@/lib/history";
import { BAND_COLOR, BAND_LABEL, bandOf, formatSeverity } from "./severity-scale";

/**
 * Past scans, from this device only. Nothing here has ever left the phone, and
 * the sheet says so — a person recording hybrid performance across someone
 * else's fields needs to know where the record lives.
 */
export function HistorySheet({
  open,
  onOpenChange,
  onReopen,
  onChanged,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onReopen: (record: ScanRecord) => void;
  onChanged: () => void;
}) {
  const [records, setRecords] = useState<ScanRecord[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    listScans()
      .then((rows) => {
        if (!alive) return;
        setRecords(rows);
        setFailed(false);
      })
      .catch(() => {
        if (!alive) return;
        setRecords([]);
        setFailed(true);
      });
    return () => {
      alive = false;
    };
  }, [open]);

  async function remove(record: ScanRecord) {
    try {
      await deleteScan(record.id);
      setRecords((prev) => prev?.filter((r) => r.id !== record.id) ?? null);
      onChanged();
      toast.success("Scan deleted");
    } catch {
      toast.error("Couldn't delete that scan");
    }
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="bottom"
        className="safe-bottom flex flex-col overflow-hidden rounded-t-2xl p-0 data-[side=bottom]:h-[86dvh]"
      >
        <SheetHeader className="shrink-0 border-b px-4 py-3.5">
          <SheetTitle className="font-display text-lg font-extrabold tracking-tight">
            Past scans
          </SheetTitle>
          <SheetDescription className="text-[0.8125rem]">
            Stored on this device only. Nothing is uploaded, and clearing your browser data removes
            them.
          </SheetDescription>
        </SheetHeader>

        <ScrollArea className="min-h-0 flex-1">
          <div className="space-y-2 p-3">
            {records === null ? (
              <>
                <Skeleton className="h-20 w-full rounded-xl" />
                <Skeleton className="h-20 w-full rounded-xl" />
                <Skeleton className="h-20 w-full rounded-xl" />
              </>
            ) : failed ? (
              <EmptyState
                title="History is unavailable"
                body="This browser blocked local storage — private browsing usually does. Scans still work, they just won't be saved."
              />
            ) : records.length === 0 ? (
              <EmptyState
                title="No scans yet"
                body="Every leaf you assess is filed here with its hybrid, location and date."
              />
            ) : (
              records.map((record) => (
                <Row key={record.id} record={record} onReopen={onReopen} onDelete={remove} />
              ))
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

function Row({
  record,
  onReopen,
  onDelete,
}: {
  record: ScanRecord;
  onReopen: (record: ScanRecord) => void;
  onDelete: (record: ScanRecord) => void;
}) {
  const band = bandOf(record.severity.percent);

  return (
    <div className="bg-card flex items-stretch gap-3 rounded-xl border p-2.5">
      <button
        type="button"
        onClick={() => onReopen(record)}
        className="focus-visible:ring-ring flex min-w-0 flex-1 items-center gap-3 rounded-lg text-left focus-visible:ring-2 focus-visible:outline-none"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={record.thumbnail}
          alt=""
          className="bg-muted size-16 shrink-0 rounded-lg object-cover"
        />
        <div className="min-w-0 flex-1">
          <p className="font-display truncate text-[0.9375rem] leading-tight font-bold">
            {record.disease.label}
          </p>
          {/* Date first: it is fixed width, so the hybrid is what truncates. */}
          <p className="text-muted-foreground eyebrow mt-1.5 truncate">
            {record.date} · {record.corn_hybrid || "No hybrid"}
          </p>
          <p className="text-muted-foreground mt-1 truncate text-[0.8125rem]">
            {record.location || "No location"}
          </p>
        </div>
        <div className="shrink-0 text-right">
          <span
            className="tabular font-display text-xl leading-none font-extrabold"
            style={{ color: BAND_COLOR[band] }}
          >
            {formatSeverity(record.severity.percent)}
            <span className="text-xs font-semibold">%</span>
          </span>
          <p className="eyebrow text-muted-foreground mt-1.5">{BAND_LABEL[band]}</p>
        </div>
      </button>
      <Button
        variant="ghost"
        onClick={() => onDelete(record)}
        aria-label={`Delete the ${record.disease.label} scan from ${record.date}`}
        className="text-muted-foreground hover:text-destructive size-11 shrink-0 self-center"
      >
        <Trash2Icon aria-hidden className="size-5" />
      </Button>
    </div>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="px-6 py-14 text-center">
      <div className="bg-secondary text-primary mx-auto grid size-14 place-items-center rounded-2xl">
        <HistoryIcon aria-hidden className="size-7" />
      </div>
      <p className="font-display mt-4 text-base font-bold">{title}</p>
      <p className="text-muted-foreground mx-auto mt-1.5 max-w-[34ch] text-sm leading-snug">
        {body}
      </p>
    </div>
  );
}
