"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { CheckIcon, LeafIcon, ScanLineIcon } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import type {
  AnalyzeDiag,
  AnalyzeResponse,
  AnalyzeSuccess,
  DiagnosisContext,
  LeafPoint,
  ScanInput,
  ScanRecord,
} from "@/lib/types";
import {
  QuotaError,
  fileToDataUri,
  flattenCutout,
  makeThumbnail,
  newId,
  recallHybrid,
  rememberHybrid,
  saveScan,
  listScans,
  hasConversation,
} from "@/lib/history";
import { NetworkError, analyze } from "./analyze";
import { CaptureCard, type Tap } from "./capture-card";
import { ChatSheet } from "./chat-sheet";
import { FieldNote } from "./field-note";
import { HistorySheet } from "./history-sheet";
import { Masthead } from "./masthead";
import { MapSheet } from "./map-sheet";
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

/** How long the marker must sit still before a speculative scan is worth firing. */
const SPECULATE_AFTER_MS = 400;

/**
 * The point as the server will actually see it.
 *
 * analyze() writes px/py at four decimals, so two points that round the same
 * produce byte-identical requests and must not invalidate each other — a drag
 * that ends a thousandth of a pixel from where it started is not a new scan.
 */
function pointKey(point: LeafPoint | null): string {
  return point ? `${point.x.toFixed(4)},${point.y.toFixed(4)}` : "none";
}

/**
 * A scan started before anyone asked for one.
 *
 * Safe because api/analyze.py never reads the field note: extract_upload()
 * returns only (bytes, filename) and analyze() takes the bytes plus the
 * ?px/py/sam query, so the answer is fixed the moment the photo and the marker
 * are settled. Firing then is real work moved earlier, not a fake head start.
 */
interface Speculation {
  /** File identity is the cache key: every new photo, including a crop, is a new File. */
  file: File;
  key: string;
  controller: AbortController;
  promise: Promise<AnalyzeResponse>;
  /** Upload progress so far, so an adopted request resumes its bar instead of restarting it. */
  fraction: number;
  uploaded: boolean;
}

export function ScanApp() {
  const [phase, setPhase] = useState<Phase>("compose");

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  // Which blade to measure: severity is lesion area over LEAF area, so the
  // point decides the denominator. CaptureCard auto-places it at the centre of
  // every new photo — the same place the server would have fallen back to — so
  // it never gates a scan; moving it is a correction, not a chore. Held here
  // rather than in CaptureCard because that card unmounts while a scan runs,
  // and the marker has to still be there when a failed scan sends the user
  // back to move it. This setter is the one place the point changes.
  const [tap, setTap] = useState<Tap | null>(null);
  const point: LeafPoint | null = tap?.point ?? null;

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
  // The id of the ScanRecord the on-screen result belongs to. The chat thread
  // is keyed to it, so it is minted before the record is written rather than
  // inside saveScan. Whether that scan already has a chat behind it — for
  // labelling the entry point "Continue chatting" — is `hasConversation(scanId)`
  // from lib/history.
  const [scanId, setScanId] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<ScanErrorCode>("internal");
  const [errorDetail, setErrorDetail] = useState<string | undefined>();
  const [errorDiag, setErrorDiag] = useState<AnalyzeDiag | undefined>();

  const [historyOpen, setHistoryOpen] = useState(false);
  const [mapOpen, setMapOpen] = useState(false);
  const [versionsOpen, setVersionsOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  // Drives the "Continue chatting" label. Refreshed when a scan is opened and
  // again when the chat sheet closes, which is the only moment the answer can
  // have changed while the result is on screen.
  const [hasChat, setHasChat] = useState(false);
  const [historyCount, setHistoryCount] = useState(0);

  const place = usePlace();
  const resultRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  // At most one speculative scan is ever alive. See Speculation above.
  const specRef = useRef<Speculation | null>(null);
  // Mirrors specRef for rendering only. The ref drives the logic — it has to,
  // because the effect must read the current speculation without re-running —
  // so this exists purely to tell the grower the work is already under way.
  const [specStatus, setSpecStatus] = useState<"idle" | "running" | "ready" | "failed">("idle");
  // Where the live speculation reports upload progress once submit() has
  // adopted it. Null while it is still running unwatched.
  const specWatcherRef = useRef<((fraction: number) => void) | null>(null);
  const specKey = pointKey(point);

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

  useEffect(
    () => () => {
      abortRef.current?.abort();
      specRef.current?.controller.abort();
    },
    [],
  );

  // Start the scan while the grower is still filling in the field note.
  //
  // The deps are a File reference and a rounded-point string, both stable
  // across re-renders, so this cannot re-fire on a render — only on a genuinely
  // new photo or a genuinely moved marker. The delay is what keeps a drag from
  // becoming a burst of invocations: pointermove churns the point, the timer
  // restarts each time, and only the position it settles on is ever sent.
  useEffect(() => {
    if (!file || phase !== "compose") return;
    // Mock runs answer from a fixture; there is nothing to get a head start on.
    if (typeof window !== "undefined" && mockCaseFrom(window.location.search)) return;

    const live = specRef.current;
    if (live && live.file === file && live.key === specKey) return;

    // The photo or the marker changed, so whatever is in flight now describes
    // pixels nobody is looking at. Kill it before it spends a serverless slot
    // on an answer that can never be adopted.
    live?.controller.abort();
    specRef.current = null;
    specWatcherRef.current = null;
    setSpecStatus("idle");

    const timer = setTimeout(() => {
      const controller = new AbortController();
      // Bounced through a mutable slot only because the record it writes to
      // cannot exist until the promise inside it does. Upload events are async,
      // so it is always wired up before the first one arrives.
      let record: Speculation | null = null;
      const promise = analyze({
        file,
        // Empty because they are not known yet and the server does not read
        // them — that is the whole reason this request can run this early.
        input: { corn_hybrid: "", location: "", date: "" },
        point,
        signal: controller.signal,
        onUploadProgress: (fraction) => {
          if (!record) return;
          record.fraction = fraction;
          if (fraction >= 1) record.uploaded = true;
          specWatcherRef.current?.(fraction);
        },
      });
      // Nobody is awaiting this yet, and an unhandled rejection would surface as
      // a console error while the grower is mid-sentence. submit() does the real
      // handling; this only marks the rejection as seen.
      promise.catch(() => {});
      record = {
        file,
        key: specKey,
        controller,
        promise,
        fraction: 0,
        uploaded: false,
      };
      specRef.current = record;
      setSpecStatus("running");
      // Only the speculation still in the slot may report — a superseded one
      // resolves after its abort and would otherwise announce a result that
      // describes the previous photo.
      void promise.then(
        (response) => {
          if (specRef.current === record) setSpecStatus(response.ok ? "ready" : "failed");
        },
        () => {
          if (specRef.current === record) setSpecStatus("failed");
        },
      );
    }, SPECULATE_AFTER_MS);

    return () => clearTimeout(timer);
  }, [file, specKey, point, phase]);

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

    // A speculation for exactly this photo and this marker is the request
    // submit() was about to make, so take it over rather than pay for it twice.
    // Anything else — none running, a different photo, a moved marker — falls
    // through to the request this has always made.
    const pending = specRef.current;
    const adopted = pending && pending.file === file && pending.key === specKey ? pending : null;
    specRef.current = null;

    if (abortRef.current && abortRef.current !== adopted?.controller) abortRef.current.abort();

    const report = (fraction: number) => {
      setUploadFraction(fraction);
      if (fraction >= 1) setUploaded(true);
    };

    /** The request as it has always been made: new controller, bar from zero. */
    const fire = () => {
      const controller = new AbortController();
      abortRef.current = controller;
      specWatcherRef.current = null;
      setUploadFraction(0);
      setUploaded(false);
      return analyze({ file, input, point, signal: controller.signal, onUploadProgress: report });
    };

    if (adopted) {
      abortRef.current = adopted.controller;
      // Pick the bar up where the upload actually is rather than restarting it
      // at zero, and take over reporting for the rest of it.
      setUploadFraction(adopted.fraction);
      setUploaded(adopted.uploaded);
      specWatcherRef.current = report;
    } else {
      setUploadFraction(0);
      setUploaded(false);
    }
    setPhase("working");

    const mockCase = typeof window !== "undefined" ? mockCaseFrom(window.location.search) : null;

    try {
      let response: AnalyzeResponse | null = null;

      if (mockCase) {
        setUploadFraction(1);
        setUploaded(true);
        response = await runMock(mockCase, file);
      } else {
        if (adopted) {
          try {
            const early = await adopted.promise;
            // `internal` means the request did not survive rather than that the
            // leaf could not be read — this function OOMs at its 1024 MB cap
            // often enough that one honest retry beats an error screen for a
            // scan the grower never watched fail. Every other outcome,
            // "no leaf detected" included, is a real reading and stands.
            if (early.ok || early.error.code !== "internal") response = early;
          } catch {
            // Cancelled by the grower (New scan) — silent, as it has always been.
            if (adopted.controller.signal.aborted) return;
          }
        }
        response ??= await fire();
      }

      if (!response.ok) {
        setErrorCode(response.error.code);
        setErrorDetail(response.error.message);
        setErrorDiag(response.error.diag);
        setPhase("error");
        return;
      }

      await showResult(response, input);
    } catch (err) {
      // An abort is the grower's own doing, so it stays silent and gets no code.
      if (err instanceof DOMException && err.name === "AbortError") return;
      setErrorCode(err instanceof NetworkError ? "network" : "internal");
      setErrorDetail(err instanceof Error ? err.message : undefined);
      setErrorDiag(
        err instanceof NetworkError
          ? err.diag
          : {
              code: "DG-CLIENT-EXC",
              reason: `${err instanceof Error ? `${err.name}: ${err.message}` : String(err)} — thrown in submit() before a response was handled.`,
            },
      );
      setPhase("error");
    }
  }

  async function showResult(response: AnalyzeSuccess, input: ScanInput) {
    const original = await fileToDataUri(file!).catch(() => null);
    // Flattened here, once, and then both shown and saved. Composing it later
    // is not an option: the record keeps no photo, so this is the last moment
    // the two halves exist together. A failure costs the picture, not the scan.
    const cutout = response.images.leaf_cutout;
    const segmented =
      original && cutout ? await flattenCutout(original, cutout).catch(() => null) : null;
    const id = newId();
    setScanId(id);
    setHasChat(false);

    setView({
      disease: response.disease,
      severity: response.severity,
      overlay: response.images.overlay,
      segmented,
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
        id,
        created_at: Date.now(),
        disease: response.disease,
        severity: response.severity,
        thumbnail,
        overlay: response.images.overlay,
        ...(segmented ? { segmented } : {}),
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
    setScanId(record.id);
    setHasChat(false);
    hasConversation(record.id)
      .then(setHasChat)
      .catch(() => setHasChat(false));
    setView({
      disease: record.disease,
      severity: record.severity,
      overlay: record.overlay,
      // Absent on records written before it was saved; the panel then shows no
      // segmentation, which is what those scans did when they were taken.
      segmented: record.segmented ?? null,
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
    specRef.current?.controller.abort();
    specRef.current = null;
    specWatcherRef.current = null;
    setSpecStatus("idle");
    setPhase("compose");
    setView(null);
    setScanId(null);
    setHasChat(false);
    setFile(null);
    setPreviewUrl(null);
    setTap(null);
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
        // Temporary demo override — see demo-healthy.ts. Keeps the assistant
        // from naming a disease the result panel declined to name.
        unclassified: view.presentAsHealthy === true,
      }
    : null;

  return (
    <>
      <Masthead
        onOpenHistory={() => setHistoryOpen(true)}
        onOpenMap={() => setMapOpen(true)}
        historyCount={historyCount}
      />

      {/* Below md this is the phone column it has always been. From 768px — iPad
          mini portrait, the narrowest tablet we are asked to serve — it splits
          in two. Widening the single column instead would put a 4:5 leaf frame
          at 976px across and 1220px tall on a 12.9", which is worse than the
          stranded column it replaces: the photo and the field note stop being
          on screen together, which is the whole point of the compose step. */}
      <main className="safe-x safe-bottom mx-auto w-full max-w-2xl [--safe-x:1rem] pt-5 pb-16 md:max-w-5xl md:[--safe-x:1.5rem] md:pt-8 md:pb-20">
        {phase === "compose" && (
          <div className="space-y-7 md:grid md:grid-cols-2 md:items-start md:gap-x-8 md:gap-y-9 md:space-y-0">
            <CaptureCard
              previewUrl={previewUrl}
              file={file}
              onSelect={selectFile}
              tap={tap}
              onTap={setTap}
              onReject={(message) => toast.error("Can't use that file", { description: message })}
            />

            <div className="space-y-7">
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
                  className="h-14 w-full gap-2.5 rounded-xl text-base font-semibold md:h-15 md:text-lg"
                >
                  <ScanLineIcon aria-hidden className="size-5 md:size-6" />
                  Assess severity
                </Button>
                {/* The scan starts as soon as the marker settles, so by the
                    time the field notes are filled in the answer is usually
                    already back. Saying so is not decoration: without it the
                    instant result looks like the app skipped the work. A
                    failed speculation deliberately says nothing — submit()
                    retries it once, and most of those succeed. */}
                <p
                  className="text-muted-foreground mt-2 flex min-h-5 items-center justify-center gap-1.5 text-center text-[0.8125rem] md:text-sm"
                  aria-live="polite"
                >
                  {!file ? (
                    "Add a leaf photo to start."
                  ) : specStatus === "running" ? (
                    <>
                      <span
                        aria-hidden
                        className="bg-primary size-1.5 shrink-0 animate-pulse rounded-full"
                      />
                      Checking this leaf now — carry on filling in the notes.
                    </>
                  ) : specStatus === "ready" ? (
                    <>
                      <CheckIcon aria-hidden className="text-primary size-3.5 shrink-0" />
                      Ready — your result will appear straight away.
                    </>
                  ) : (
                    "Takes about 2–5 seconds."
                  )}
                </p>
              </div>
            </div>

            <div className="md:col-span-2">
              <Provenance onOpenVersions={() => setVersionsOpen(true)} />
            </div>
          </div>
        )}

        <div ref={resultRef} className="scroll-mt-20 md:scroll-mt-24">
          {phase === "working" && (
            <div className="space-y-4 md:grid md:grid-cols-2 md:items-start md:gap-6 md:space-y-0">
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
            // One card of prose and a short list — it does not want 976px, and
            // splitting instructions into columns would break their order.
            <div className="md:mx-auto md:max-w-2xl">
              <ScanError
                code={errorCode}
                detail={errorDetail}
                diag={errorDiag}
                onRetake={newScan}
                onRetry={submit}
                // Compose still holds the photo and the marker, so this is a
                // re-tap rather than a restart.
                onAdjust={() => setPhase("compose")}
              />
            </div>
          )}

          {phase === "result" && view && (
            <ResultPanel
              view={view}
              onAskAssistant={() => setChatOpen(true)}
              hasConversation={hasChat}
              onNewScan={newScan}
            />
          )}
        </div>
      </main>

      <MapSheet open={mapOpen} onOpenChange={setMapOpen} onReopen={reopen} />

      <HistorySheet
        open={historyOpen}
        onOpenChange={setHistoryOpen}
        onReopen={reopen}
        onChanged={refreshCount}
      />
      <ChatSheet
        open={chatOpen}
        onOpenChange={(open) => {
          setChatOpen(open);
          if (!open && scanId) {
            hasConversation(scanId)
              .then(setHasChat)
              .catch(() => setHasChat(false));
          }
        }}
        context={context}
        scanId={scanId}
      />
      <VersionSheet open={versionsOpen} onOpenChange={setVersionsOpen} />
    </>
  );
}

function Provenance({ onOpenVersions }: { onOpenVersions: () => void }) {
  return (
    // Capped in ch, not by the grid: spanning both columns keeps it out of the
    // action's way, but 976px of 13px prose is an unreadable measure.
    <div className="text-muted-foreground border-t pt-5 text-[0.8125rem] leading-relaxed md:max-w-[72ch] md:text-sm">
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
