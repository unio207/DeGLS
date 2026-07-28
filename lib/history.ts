/**
 * Local-only scan history.
 *
 * Everything here stays on the device: IndexedDB, no server, no auth, no sync.
 * Overlays come back from the API as full-size data URIs, which will blow past
 * the origin's storage quota within a couple of dozen scans, so the list view
 * only ever reads a downscaled JPEG thumbnail. The full overlay is stored on
 * the same record but read lazily when a scan is reopened.
 */

import type { ScanRecord } from "@/lib/types";

const DB_NAME = "degls";
const DB_VERSION = 1;
const STORE = "scans";
const HYBRID_KEY = "degls.last-hybrid";

export class QuotaError extends Error {
  constructor() {
    super("Device storage is full.");
    this.name = "QuotaError";
  }
}

function isQuotaError(err: unknown): boolean {
  if (typeof DOMException !== "undefined" && err instanceof DOMException) {
    return (
      err.name === "QuotaExceededError" ||
      err.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
      err.code === 22
    );
  }
  return false;
}

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (typeof indexedDB === "undefined") {
    return Promise.reject(new Error("IndexedDB is unavailable."));
  }
  if (!dbPromise) {
    dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          const store = db.createObjectStore(STORE, { keyPath: "id" });
          store.createIndex("created_at", "created_at");
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error ?? new Error("Could not open the history database."));
      req.onblocked = () => reject(new Error("History database is blocked by another tab."));
    }).catch((err) => {
      dbPromise = null;
      throw err;
    });
  }
  return dbPromise;
}

function tx<T>(
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        let transaction: IDBTransaction;
        try {
          transaction = db.transaction(STORE, mode);
        } catch (err) {
          reject(err);
          return;
        }
        const request = run(transaction.objectStore(STORE));
        request.onsuccess = () => resolve(request.result);
        transaction.onerror = () => {
          const err = transaction.error ?? request.error;
          reject(isQuotaError(err) ? new QuotaError() : (err ?? new Error("History write failed.")));
        };
        transaction.onabort = () => {
          const err = transaction.error;
          reject(isQuotaError(err) ? new QuotaError() : (err ?? new Error("History write aborted.")));
        };
      }),
  );
}

/** Newest first. Thumbnails only — `overlay` is stripped to keep the list light. */
export async function listScans(limit = 60): Promise<ScanRecord[]> {
  const all = await tx<ScanRecord[]>("readonly", (store) => store.getAll() as IDBRequest<ScanRecord[]>);
  return all.sort((a, b) => b.created_at - a.created_at).slice(0, limit);
}

export async function getScan(id: string): Promise<ScanRecord | undefined> {
  return tx<ScanRecord | undefined>("readonly", (store) => store.get(id) as IDBRequest<ScanRecord | undefined>);
}

/**
 * Saves a scan. On a quota failure it drops the three oldest records and
 * retries once; if that still fails the caller gets a QuotaError to surface.
 */
export async function saveScan(record: ScanRecord): Promise<void> {
  try {
    await tx("readwrite", (store) => store.put(record) as IDBRequest<IDBValidKey>);
    return;
  } catch (err) {
    if (!(err instanceof QuotaError)) throw err;
  }

  const existing = await listScans(1000);
  const oldest = existing.slice(-3).map((s) => s.id);
  for (const id of oldest) {
    try {
      await deleteScan(id);
    } catch {
      /* best effort */
    }
  }
  await tx("readwrite", (store) => store.put(record) as IDBRequest<IDBValidKey>);
}

export async function deleteScan(id: string): Promise<void> {
  await tx("readwrite", (store) => store.delete(id) as IDBRequest<undefined>);
}

export async function clearScans(): Promise<void> {
  await tx("readwrite", (store) => store.clear() as IDBRequest<undefined>);
}

/**
 * Hybrid names repeat across a whole field season, so the last one used is
 * pre-filled on the next scan. Kept in localStorage rather than IndexedDB so
 * the form can render it synchronously without a flash of empty input.
 */
export function rememberHybrid(hybrid: string): void {
  const value = hybrid.trim();
  if (!value) return;
  try {
    localStorage.setItem(HYBRID_KEY, value);
  } catch {
    /* private mode — not worth surfacing */
  }
}

export function recallHybrid(): string {
  try {
    return localStorage.getItem(HYBRID_KEY) ?? "";
  } catch {
    return "";
  }
}

/** Downscales an image data URI to a JPEG thumbnail no wider/taller than `max`. */
export async function makeThumbnail(dataUri: string, max = 320, quality = 0.7): Promise<string> {
  const img = await loadImage(dataUri);
  const scale = Math.min(1, max / Math.max(img.width, img.height));
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return dataUri;
  ctx.drawImage(img, 0, 0, w, h);
  return canvas.toDataURL("image/jpeg", quality);
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not read that image."));
    img.src = src;
  });
}

export function fileToDataUri(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read that file."));
    reader.readAsDataURL(file);
  });
}

export function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
