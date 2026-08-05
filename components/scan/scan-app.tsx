"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { LeafIcon, ScanLineIcon } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import type { AnalyzeSuccess, DiagnosisContext, ScanInput, ScanRecord } from "@/lib/types";
import {
  QuotaError,
  fileToDataUri,
  makeThumbnail,
  newId,
  recallHybrid,
  rememberHybrid,
  saveScan,
  listScans,
} from "@/lib/history";
import { NetworkError, analyze } from "./analyze";
import { CaptureCard } from "./capture-card";
import { ChatSheet } from "./chat-sheet";
import { FieldNote } from "./field-note";
import { HistorySheet } from "./history-sheet";
import { Masthead } from "./masthead";
import { CURRENT_VERSION, VersionSheet } from "./version-sheet";
import { ProgressStages } from "./progress-stages";
import { ResultPanel, type ResultView } from "./result-panel";
import { ScanError, type ScanErrorCode } from "./scan-error";
import { isDemoHealthyHybrid } from "./demo-healthy";
import { mockCaseFrom, runMock } from "./mock";
import { usePlace, type Fix } from "./use-place";

type Phase = "compose" | "working" | "result" | "error";

/** localStorage is read once at mount and never changes underneath us. */
function subscribeNever(): () => void {
  return () => {};
}

function today(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function ScanApp() {
  const [phase, setPhase] = useState<Phase>("compose");

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // The last hybrid used is read straight out of localStorage rather than
  // being copied into state by an effect, so the input never flashes empty on
  // the first paint. Typing takes over from there.
  const rememberedHybrid = useSyncExternalStore(subscribeNever, recallHybrid, () => "");
  const [typedHybrid, setTypedHybrid] = useState<string | null>(null);
  const hybrid = typedHybrid ?? rememberedHybrid;
  const setHybrid = setTypedHybrid;

  const [location, setLocation] = useState("");
  const [fix, setFix] = useState<Fix | null>(null);

  // "Today" is the device's today, not the build machine's — the page is
  // statically prerendered, so reading the clock during render would bake a
  // stale date into the HTML and mismatch on hydration.
  const todayOnDevice = useSyncExternalStore(subscribeNever, today, () => "");
  const [pickedDate, setPickedDate] = useState<string | null>(null);
  const date = pickedDate ?? todayOnDevice;
  const setDate = setPickedDate;

  const [uploadFraction, setUploadFraction] = useState(0);
  const [uploaded, setUploaded] = useState(false);

  const [view, setView] = useState<ResultView | null>(null);
  const [errorCode, setErrorCode] = useState<ScanErrorCode>("internal");
  const [errorDetail, setErrorDetail] = useState<string | undefined>();

  const [historyOpen, setHistoryOpen] = useState(false);
  const [versionsOpen, setVersionsOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [historyCount, setHistoryCount] = useState(0);

  const place = usePlace();
  const resultRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const refreshCount = useCallback(() => {
    listScans(1000)
      .then((r) => setHistoryCount(r.length))
      .catch(() => setHistoryCount(0));
  }, []);

  useEffect(refreshCount, [refreshCount]);

  useEffect(() => {
    if (!previewUrl?.startsWith("blob:")) return;
    return () => URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  useEffect(() => () => abortRef.current?.abort(), []);

  function selectFile(next: File) {
    setFile(next);
    setPreviewUrl(URL.createObjectURL(next));
    if (phase !== "compose") setPhase("compose");
  }

  /** Location is stored as the place name with the raw fix kept alongside it. */
  function composedLocation(): string {
    if (fix && location.trim()) {
      return `${location.trim()} (${fix.lat.toFixed(4)}, ${fix.lon.toFixed(4)})`;
    }
    return location.trim();
  }

  async function locate() {
    const next = await place.locate();
    if (!next) return;
    setFix(next);
    setLocation(next.label);
  }

  async function submit() {
    if (!file) return;

    const input: ScanInput = {
      corn_hybrid: hybrid.trim(),
      location: composedLocation(),
      date,
    };

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setUploadFraction(0);
    setUploaded(false);
    setPhase("working");

    const mockCase = typeof window !== "undefined" ? mockCaseFrom(window.location.search) : null;

    try {
      let response;
      if (mockCase) {
        setUploadFraction(1);
        setUploaded(true);
        response = await runMock(mockCase, file);
      } else {
        response = await analyze({
          file,
          input,
          signal: controller.signal,
          onUploadProgress: (fraction) => {
            setUploadFraction(fraction);
            if (fraction >= 1) setUploaded(true);
          },
        });
      }

      if (!response.ok) {
        setErrorCode(response.error.code);
        setErrorDetail(response.error.message);
        setPhase("error");
        return;
      }

      await showResult(response, input);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setErrorCode(err instanceof NetworkError ? "network" : "internal");
      setErrorDetail(err instanceof Error ? err.message : undefined);
      setPhase("error");
    }
  }

  async function showResult(response: AnalyzeSuccess, input: ScanInput) {
    const original = await fileToDataUri(file!).catch(() => null);

    setView({
      disease: response.disease,
      severity: response.severity,
      overlay: response.images.overlay,
      original,
      input,
      meta: response.meta,
      // Temporary demo override — see demo-healthy.ts. Remove after the demo.
      presentAsHealthy: isDemoHealthyHybrid(input.corn_hybrid),
    });
    setPhase("result");
    rememberHybrid(input.corn_hybrid);

    // History is written after the result is on screen — saving must never be
    // the thing standing between a scan and its answer.
    try {
      const thumbnail = await makeThumbnail(response.images.overlay);
      const record: ScanRecord = {
        ...input,
        id: newId(),
        created_at: Date.now(),
        disease: response.disease,
        severity: response.severity,
        thumbnail,
        overlay: response.images.overlay,
      };
      await saveScan(record);
      refreshCount();
    } catch (err) {
      if (err instanceof QuotaError) {
        toast.error("Storage is full", {
          description: "Delete some past scans to keep saving new ones.",
          action: { label: "Open history", onClick: () => setHistoryOpen(true) },
        });
      } else {
        toast.error("This scan wasn't saved to history", {
          description: "The result on screen is still good.",
        });
      }
    }
  }

  // Bring the answer into view without yanking the page around mid-render.
  useEffect(() => {
    if (phase !== "result" && phase !== "error") return;
    const id = requestAnimationFrame(() => {
      resultRef.current?.scrollIntoView({ block: "start", behavior: "smooth" });
    });
    return () => cancelAnimationFrame(id);
  }, [phase]);

  function reopen(record: ScanRecord) {
    setHistoryOpen(false);
    setView({
      disease: record.disease,
      severity: record.severity,
      overlay: record.overlay,
      original: null,
      input: {
        corn_hybrid: record.corn_hybrid,
        location: record.location,
        date: record.date,
      },
      meta: null,
      recordedAt: record.created_at,
      // Same temporary demo override, so reopening the scan from history shows
      // what the live scan showed. See demo-healthy.ts; remove after the demo.
      presentAsHealthy: isDemoHealthyHybrid(record.corn_hybrid),
    });
    setPhase("result");
  }

  function newScan() {
    abortRef.current?.abort();
    setPhase("compose");
    setView(null);
    setFile(null);
    setPreviewUrl(null);
    setUploadFraction(0);
    setUploaded(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const context: DiagnosisContext | null = view
    ? {
        ...view.input,
        disease_label: view.disease.label,
        disease_code: view.disease.code,
        severity_percent: view.severity.percent,
        confidence: view.disease.confidence,
      }
    : null;

  return (
    <>
      <Masthead onOpenHistory={() => setHistoryOpen(true)} historyCount={historyCount} />

      <main className="safe-x safe-bottom mx-auto w-full max-w-2xl px-4 pt-5 pb-16">
        {phase === "compose" && (
          <div className="space-y-7">
            <CaptureCard
              previewUrl={previewUrl}
              onSelect={selectFile}
              onReject={(message) => toast.error("Can't use that file", { description: message })}
            />

            <FieldNote
              hybrid={hybrid}
              onHybridChange={setHybrid}
              location={location}
              onLocationChange={setLocation}
              fix={fix}
              onClearFix={() => {
                setFix(null);
                place.reset();
              }}
              date={date}
              onDateChange={setDate}
              onLocate={locate}
              placeStatus={place.status}
              placeMessage={place.message}
            />

            <div>
              <Button
                onClick={submit}
                disabled={!file}
                className="h-14 w-full gap-2.5 rounded-xl text-base font-semibold"
              >
                <ScanLineIcon aria-hidden className="size-5" />
                Assess severity
              </Button>
              <p
                className="text-muted-foreground mt-2 min-h-5 text-center text-[0.8125rem]"
                aria-live="polite"
              >
                {file ? "Takes about 2–5 seconds." : "Add a leaf photo to start."}
              </p>
            </div>

            <Provenance onOpenVersions={() => setVersionsOpen(true)} />
          </div>
        )}

        <div ref={resultRef} className="scroll-mt-20">
          {phase === "working" && (
            <div className="space-y-4">
              {previewUrl && (
                <div className="bg-muted relative aspect-[4/5] w-full overflow-hidden rounded-2xl">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewUrl}
                    alt="The leaf photo being analysed"
                    className="absolute inset-0 h-full w-full object-cover opacity-70"
                  />
                  <div className="absolute inset-0 bg-black/15" />
                </div>
              )}
              <ProgressStages uploadFraction={uploadFraction} uploaded={uploaded} />
            </div>
          )}

          {phase === "error" && (
            <ScanError
              code={errorCode}
              detail={errorDetail}
              onRetake={newScan}
              onRetry={submit}
            />
          )}

          {phase === "result" && view && (
            <ResultPanel
              view={view}
              onAskAssistant={() => setChatOpen(true)}
              onNewScan={newScan}
            />
          )}
        </div>
      </main>

      <HistorySheet
        open={historyOpen}
        onOpenChange={setHistoryOpen}
        onReopen={reopen}
        onChanged={refreshCount}
      />
      <ChatSheet open={chatOpen} onOpenChange={setChatOpen} context={context} />
      <VersionSheet open={versionsOpen} onOpenChange={setVersionsOpen} />
    </>
  );
}

function Provenance({ onOpenVersions }: { onOpenVersions: () => void }) {
  return (
    <div className="text-muted-foreground border-t pt-5 text-[0.8125rem] leading-relaxed">
      <p className="flex gap-2">
        <LeafIcon aria-hidden className="mt-0.5 size-4 shrink-0" />
        <span>
          DeGLS reads gray leaf spot, northern leaf blight and common rust from a single leaf photo
          and measures how much of the blade is lesioned. Photos and scan history stay on this
          device.
        </span>
      </p>
      <button
        type="button"
        onClick={onOpenVersions}
        className="hover:text-foreground mt-3 -mx-1 px-1 py-1 text-[0.75rem] underline underline-offset-2 transition-colors"
      >
        Version history
        <span className="tabular"> · v{CURRENT_VERSION}</span>
      </button>
    </div>
  );
}
