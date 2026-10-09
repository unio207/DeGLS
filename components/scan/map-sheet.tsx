"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  CrosshairIcon,
  MapPinnedIcon,
  MinusIcon,
  PlusIcon,
  WifiOffIcon,
  XIcon,
} from "lucide-react";
import type { Map as LeafletMap, Marker } from "leaflet";
import "leaflet/dist/leaflet.css";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import type { ScanRecord } from "@/lib/types";
import { listScans } from "@/lib/history";
import { BAND_COLOR, BAND_LABEL, BAND_ORDER, bandOf, formatSeverity } from "./severity-scale";
import { isDemoHealthyHybrid } from "./demo-healthy";
import { pinsFrom, type ScanPin } from "./scan-coords";

/**
 * Where past scans were taken.
 *
 * Rendering is Leaflet over OpenStreetMap raster tiles: keyless, no account, no
 * env var. Leaflet is here for the touch handling — pinch-zoom, inertial pan and
 * fit-to-bounds that behave correctly on a phone are the part that is genuinely
 * hard to write, not the projection.
 *
 * The demo happens in a field, so tiles are treated as optional decoration. The
 * container sits on a ruled backdrop instead of Leaflet's default flat grey, the
 * pins are positioned by arithmetic that needs no network, and a scale bar gives
 * the spacing between them a real distance. With the tile host blocked the map
 * degrades to an accurate, legible scatter of pins rather than a broken box.
 */
export function MapSheet({
  open,
  onOpenChange,
  onReopen,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Reopening the scan is the caller's job; the sheet closes itself first. */
  onReopen: (record: ScanRecord) => void;
}) {
  const [records, setRecords] = useState<ScanRecord[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

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

  const pins = useMemo(() => (records ? pinsFrom(records) : []), [records]);
  const selected = pins.find((pin) => pin.record.id === selectedId) ?? null;

  // Closing always drops the selection, so reopening the sheet does not come
  // back with a card floating over a map that has not been fitted yet.
  function setOpen(next: boolean) {
    if (!next) setSelectedId(null);
    onOpenChange(next);
  }

  function reopen(record: ScanRecord) {
    setOpen(false);
    onReopen(record);
  }

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetContent
        side="bottom"
        className="safe-bottom mx-auto flex flex-col overflow-hidden rounded-t-2xl p-0 data-[side=bottom]:h-[86dvh] md:max-w-3xl md:rounded-t-3xl md:data-[side=bottom]:h-[82dvh]"
      >
        <SheetHeader className="shrink-0 border-b px-4 py-3.5 md:px-5 md:py-4">
          <SheetTitle className="font-display text-lg font-extrabold tracking-tight md:text-xl">
            Scan map
          </SheetTitle>
          <SheetDescription className="text-[0.8125rem] md:text-sm">
            Every saved scan that was taken with the locate button, coloured by severity band. Tap a
            pin to open that scan.
          </SheetDescription>
        </SheetHeader>

        <div className="relative min-h-0 flex-1">
          {records === null ? (
            <div className="absolute inset-0 p-3">
              <Skeleton className="size-full rounded-xl" />
            </div>
          ) : failed ? (
            <EmptyState
              title="History is unavailable"
              body="This browser blocked local storage — private browsing usually does. Scans still work, they just aren't saved, so there is nothing to map."
            />
          ) : records.length === 0 ? (
            <EmptyState
              title="No scans yet"
              body="Once you assess a leaf it is filed with its hybrid, location and date, and shows up here."
            />
          ) : pins.length === 0 ? (
            <EmptyState
              title="No scan has a position"
              body="A typed field name can't be placed on a map. Tap the locate button next to Location when you record a scan and it will appear here."
            />
          ) : (
            <PinMap pins={pins} selectedId={selectedId} onSelect={setSelectedId} />
          )}

          {selected && (
            <SelectedCard pin={selected} onOpen={reopen} onDismiss={() => setSelectedId(null)} />
          )}
        </div>

        {/* No pins, no map: a severity key and a tile credit for nothing on
            screen is just noise. */}
        {pins.length > 0 && (
          <div className="flex shrink-0 flex-wrap items-center justify-between gap-x-4 gap-y-2 border-t px-4 py-2.5">
            <ul className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
              {BAND_ORDER.map((band) => (
                <li key={band} className="flex items-center gap-1.5">
                  <span
                    className="size-2.5 shrink-0 rounded-full"
                    style={{ background: BAND_COLOR[band] }}
                    aria-hidden
                  />
                  <span className="eyebrow text-muted-foreground">{BAND_LABEL[band]}</span>
                </li>
              ))}
            </ul>
            {/* Attribution lives out here rather than over the map so a selected
              scan card can never cover it. */}
            <a
              href="https://www.openstreetmap.org/copyright"
              target="_blank"
              rel="noreferrer"
              className="eyebrow text-muted-foreground hover:text-foreground shrink-0"
            >
              © OpenStreetMap
            </a>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}

/* ------------------------------------------------------------------ map ---- */

const TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

/** One stray 404 is normal at the edge of coverage; a run of them is no signal. */
const TILE_FAILURE_THRESHOLD = 3;

/** Close enough to read a field boundary, far enough that one pin isn't a void. */
const SINGLE_PIN_ZOOM = 14;

/**
 * Ruled backdrop behind the tile pane. It does not pan with the map — it is not
 * a graticule and does not pretend to be — but it means a tile-less map reads as
 * a plotting surface rather than a failure.
 */
const BACKDROP: CSSProperties = {
  backgroundColor: "var(--muted)",
  backgroundImage:
    "linear-gradient(to right, color-mix(in oklch, var(--muted-foreground) 14%, transparent) 1px, transparent 1px), linear-gradient(to bottom, color-mix(in oklch, var(--muted-foreground) 14%, transparent) 1px, transparent 1px)",
  backgroundSize: "36px 36px",
};

/**
 * Leaflet ships its own look. Only the parts that would read as foreign in this
 * app are overridden: the default grey fill (which is the "broken map" box), the
 * font, and the scale bar's colours.
 */
const MAP_CSS = `
.degls-map .leaflet-container { background: transparent; font: inherit; outline: none; }
.degls-map .leaflet-control-scale-line {
  background: color-mix(in oklch, var(--background) 82%, transparent);
  border: 1px solid var(--border); border-top: none;
  color: var(--muted-foreground); font-family: var(--font-mono); font-size: 10px;
  padding: 1px 5px; box-shadow: none;
}
.degls-pin {
  display: block; width: 100%; height: 100%; border-radius: 9999px;
  border: 2px solid var(--background); box-shadow: 0 1px 3px rgb(0 0 0 / 0.45);
  transition: transform 120ms ease;
}
.degls-map .leaflet-marker-icon { cursor: pointer; }
.degls-map .leaflet-marker-icon[data-selected="true"] .degls-pin {
  transform: scale(1.45);
  box-shadow: 0 0 0 3px color-mix(in oklch, var(--foreground) 45%, transparent), 0 2px 6px rgb(0 0 0 / 0.5);
}
.degls-map .leaflet-marker-icon:focus-visible { outline: none; }
.degls-map .leaflet-marker-icon:focus-visible .degls-pin { transform: scale(1.45); box-shadow: 0 0 0 3px var(--ring); }
`;

function pinColor(record: ScanRecord): string {
  // Same temporary demo override as the history list, recomputed rather than
  // stored so the map cannot disagree with the result panel. See demo-healthy.ts.
  if (isDemoHealthyHybrid(record.corn_hybrid)) return "var(--muted-foreground)";
  return BAND_COLOR[bandOf(record.severity.percent)];
}

function pinLabel(record: ScanRecord): string {
  return isDemoHealthyHybrid(record.corn_hybrid) ? "Healthy" : record.disease.label;
}

function PinMap({
  pins,
  selectedId,
  onSelect,
}: {
  pins: ScanPin[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const markersRef = useRef(new Map<string, Marker>());
  const fitRef = useRef<() => void>(() => {});
  const touchedRef = useRef(false);
  const [ready, setReady] = useState(false);
  const [tilesOk, setTilesOk] = useState(true);

  // `onSelect` is a setState function from the parent, so it is stable and this
  // does not rebuild the map on every render.
  useEffect(() => {
    let cancelled = false;
    const host = hostRef.current;
    const markers = markersRef.current;

    (async () => {
      // Leaflet touches `window` on import, so it is only ever pulled in here —
      // an effect never runs on the server, which keeps prerendering intact
      // without a next/dynamic wrapper.
      const L = await import("leaflet");
      if (cancelled || !host) return;

      const map = L.map(host, {
        zoomControl: false,
        attributionControl: false,
      });
      mapRef.current = map;

      let tileErrors = 0;
      const tiles = L.tileLayer(TILE_URL, { maxZoom: 19, crossOrigin: true });
      tiles.on("tileload", () => {
        tileErrors = 0;
        if (!cancelled) setTilesOk(true);
      });
      tiles.on("tileerror", () => {
        tileErrors += 1;
        if (!cancelled && tileErrors >= TILE_FAILURE_THRESHOLD) setTilesOk(false);
      });
      tiles.addTo(map);

      // Imperial only: this is a US corn tool and two bars is clutter on a phone.
      L.control.scale({ metric: false, imperial: true, position: "bottomleft" }).addTo(map);

      const bounds = L.latLngBounds([]);
      for (const pin of pins) {
        const marker = L.marker([pin.lat, pin.lon], {
          keyboard: true,
          riseOnHover: true,
          title: `${pinLabel(pin.record)} · ${pin.record.date}`,
          icon: L.divIcon({
            className: "",
            iconSize: [22, 22],
            iconAnchor: [11, 11],
            html: `<span class="degls-pin" style="background:${pinColor(pin.record)}"></span>`,
          }),
        });
        marker.on("click", () => onSelect(pin.record.id));
        marker.addTo(map);
        markers.set(pin.record.id, marker);
        bounds.extend([pin.lat, pin.lon]);
      }

      // A single pin — or several from the same field — has no extent to fit, so
      // fitBounds would slam to max zoom on a point. Pick a readable zoom instead.
      fitRef.current = () => {
        const current = mapRef.current;
        if (!current) return;
        if (pins.length === 1) {
          current.setView([pins[0].lat, pins[0].lon], SINGLE_PIN_ZOOM);
        } else {
          current.fitBounds(bounds, {
            padding: [48, 48],
            maxZoom: SINGLE_PIN_ZOOM + 1,
          });
        }
      };
      fitRef.current();
      setReady(true);
    })();

    return () => {
      cancelled = true;
      mapRef.current?.remove();
      mapRef.current = null;
      markers.clear();
      setReady(false);
    };
  }, [pins, onSelect]);

  // The sheet animates in and an iPad rotates, both of which resize the host
  // after Leaflet has measured it — and a resized viewport keeps the centre but
  // not the fit, so the pins drift off to one side. Refit until the grower has
  // moved the map themselves, after which their view is theirs to keep.
  useEffect(() => {
    const host = hostRef.current;
    if (!host || !ready) return;
    const remeasure = () => {
      mapRef.current?.invalidateSize();
      if (!touchedRef.current) fitRef.current();
    };
    // Both, deliberately: the observer catches the sheet settling and a split
    // view resizing, the window events catch an iPad rotating.
    const observer = new ResizeObserver(remeasure);
    observer.observe(host);
    window.addEventListener("resize", remeasure);
    window.addEventListener("orientationchange", remeasure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", remeasure);
      window.removeEventListener("orientationchange", remeasure);
    };
  }, [ready]);

  useEffect(() => {
    if (!ready) return;
    for (const [id, marker] of markersRef.current) {
      const el = marker.getElement();
      if (el) el.dataset.selected = String(id === selectedId);
      marker.setZIndexOffset(id === selectedId ? 1000 : 0);
    }
    const pin = pins.find((p) => p.record.id === selectedId);
    // Keep the selected pin clear of the card that just covered the bottom.
    if (pin) mapRef.current?.panInside([pin.lat, pin.lon], { padding: [48, 48] });
  }, [selectedId, pins, ready]);

  return (
    <div className="degls-map relative size-full overflow-hidden">
      <style>{MAP_CSS}</style>
      <div className="absolute inset-0" style={BACKDROP} aria-hidden />
      <div
        ref={hostRef}
        className="absolute inset-0"
        // Any drag, pinch or wheel over the map counts as taking control, which
        // is the one thing Leaflet's own events cannot tell apart from a
        // programmatic move.
        onPointerDownCapture={() => {
          touchedRef.current = true;
        }}
        onWheelCapture={() => {
          touchedRef.current = true;
        }}
      />

      <div className="absolute top-3 right-3 z-[900] flex flex-col gap-2">
        <MapButton
          label="Zoom in"
          onClick={() => {
            touchedRef.current = true;
            mapRef.current?.zoomIn();
          }}
        >
          <PlusIcon aria-hidden className="size-5" />
        </MapButton>
        <MapButton
          label="Zoom out"
          onClick={() => {
            touchedRef.current = true;
            mapRef.current?.zoomOut();
          }}
        >
          <MinusIcon aria-hidden className="size-5" />
        </MapButton>
        <MapButton
          label="Fit all pins"
          onClick={() => {
            touchedRef.current = false;
            fitRef.current();
          }}
        >
          <CrosshairIcon aria-hidden className="size-5" />
        </MapButton>
      </div>

      {!tilesOk && (
        <div className="bg-card/90 absolute top-3 left-3 z-[900] flex max-w-[22rem] items-start gap-2 rounded-lg border px-2.5 py-2 shadow-sm backdrop-blur-sm">
          <WifiOffIcon aria-hidden className="text-muted-foreground mt-px size-4 shrink-0" />
          <p className="text-[0.8125rem] leading-snug">
            No signal for map imagery. The pins are still in the right places — the bar at the
            bottom left is the scale.
          </p>
        </div>
      )}
    </div>
  );
}

function MapButton({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <Button
      variant="ghost"
      onClick={onClick}
      aria-label={label}
      // Not the outline variant: its dark-mode fill is translucent, which over
      // map imagery leaves the glyph unreadable.
      className="border-border bg-background/90 text-foreground hover:bg-background size-11 border shadow-sm backdrop-blur-sm md:size-12"
    >
      {children}
    </Button>
  );
}

/* -------------------------------------------------------------- selected ---- */

function SelectedCard({
  pin,
  onOpen,
  onDismiss,
}: {
  pin: ScanPin;
  onOpen: (record: ScanRecord) => void;
  onDismiss: () => void;
}) {
  const { record } = pin;
  const band = bandOf(record.severity.percent);
  const healthy = isDemoHealthyHybrid(record.corn_hybrid);

  return (
    <div className="bg-card/95 absolute inset-x-3 bottom-3 z-[900] rounded-xl border p-3 shadow-lg backdrop-blur">
      <div className="flex items-start gap-3">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={record.thumbnail}
          alt=""
          className="bg-muted size-14 shrink-0 rounded-lg object-cover md:size-16"
        />
        <div className="min-w-0 flex-1">
          <p className="font-display truncate text-[0.9375rem] leading-tight font-bold">
            {healthy ? "Healthy" : record.disease.label}
          </p>
          <p className="text-muted-foreground eyebrow mt-1.5 truncate">
            {record.date} · {record.corn_hybrid || "No hybrid"}
          </p>
          <p className="text-muted-foreground mt-1 truncate text-[0.8125rem]">
            {pin.place || `${pin.lat.toFixed(4)}, ${pin.lon.toFixed(4)}`}
          </p>
        </div>
        <div className="shrink-0 text-right">
          {healthy ? (
            <span className="text-muted-foreground eyebrow">Not classified</span>
          ) : (
            <>
              <span
                className="tabular font-display text-xl leading-none font-extrabold"
                style={{ color: BAND_COLOR[band] }}
              >
                {formatSeverity(record.severity.percent)}
                <span className="text-xs font-semibold">%</span>
              </span>
              <p className="eyebrow text-muted-foreground mt-1.5">{BAND_LABEL[band]}</p>
            </>
          )}
        </div>
      </div>
      <div className="mt-2.5 flex gap-2">
        <Button onClick={() => onOpen(record)} className="h-11 flex-1 md:h-12">
          Open this scan
        </Button>
        <Button
          variant="outline"
          onClick={onDismiss}
          aria-label="Dismiss"
          className="size-11 shrink-0 md:size-12"
        >
          <XIcon aria-hidden className="size-5" />
        </Button>
      </div>
    </div>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="grid h-full place-items-center px-6 py-10 text-center">
      <div>
        <div className="bg-secondary text-primary mx-auto grid size-14 place-items-center rounded-2xl">
          <MapPinnedIcon aria-hidden className="size-7" />
        </div>
        <p className="font-display mt-4 text-base font-bold">{title}</p>
        <p className="text-muted-foreground mx-auto mt-1.5 max-w-[36ch] text-sm leading-snug">
          {body}
        </p>
      </div>
    </div>
  );
}
