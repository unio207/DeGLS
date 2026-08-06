/**
 * Local-only scan history.
 *
 * Everything here stays on the device: IndexedDB, no server, no auth, no sync.
 * Overlays come back from the API as full-size data URIs, which will blow past
 * the origin's storage quota within a couple of dozen scans, so the list view
 * only ever reads a downscaled JPEG thumbnail. The full overlay is stored on
 * the same record but read lazily when a scan is reopened.
 */

import type { ConversationRecord, DiagnosisUIMessage, ScanRecord } from "@/lib/types";

const DB_NAME = "degls";
// v2 adds the `conversations` store. Existing v1 databases upgrade in place —
// scans are untouched and there is nothing to backfill, since no conversation
// was ever persisted before this version.
const DB_VERSION = 2;
const STORE = "scans";
const CONVERSATIONS = "conversations";
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
        if (!db.objectStoreNames.contains(CONVERSATIONS)) {
          db.createObjectStore(CONVERSATIONS, { keyPath: "scan_id" });
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
  storeName: string,
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        let transaction: IDBTransaction;
        try {
          transaction = db.transaction(storeName, mode);
        } catch (err) {
          reject(err);
          return;
        }
        const request = run(transaction.objectStore(storeName));
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
  const all = await tx<ScanRecord[]>(STORE, "readonly", (store) => store.getAll() as IDBRequest<ScanRecord[]>);
  return all.sort((a, b) => b.created_at - a.created_at).slice(0, limit);
}

export async function getScan(id: string): Promise<ScanRecord | undefined> {
  return tx<ScanRecord | undefined>(STORE, "readonly", (store) => store.get(id) as IDBRequest<ScanRecord | undefined>);
}

/**
 * Saves a scan. On a quota failure it drops the three oldest records and
 * retries once; if that still fails the caller gets a QuotaError to surface.
 */
export async function saveScan(record: ScanRecord): Promise<void> {
  try {
    await tx(STORE, "readwrite", (store) => store.put(record) as IDBRequest<IDBValidKey>);
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
  await tx(STORE, "readwrite", (store) => store.put(record) as IDBRequest<IDBValidKey>);
}

export async function deleteScan(id: string): Promise<void> {
  await tx(STORE, "readwrite", (store) => store.delete(id) as IDBRequest<undefined>);
  // A conversation with no scan behind it is unreachable, so it goes with it.
  await deleteConversation(id).catch(() => {});
}

export async function clearScans(): Promise<void> {
  await tx(STORE, "readwrite", (store) => store.clear() as IDBRequest<undefined>);
  await tx(CONVERSATIONS, "readwrite", (store) => store.clear() as IDBRequest<undefined>);
}

/* ---------------------------------------------------------------- chat ---- */

/** The saved thread for a scan, or `undefined` if the grower never asked anything. */
export async function getConversation(scanId: string): Promise<ConversationRecord | undefined> {
  return tx<ConversationRecord | undefined>(
    CONVERSATIONS,
    "readonly",
    (store) => store.get(scanId) as IDBRequest<ConversationRecord | undefined>,
  );
}

/**
 * Whether a scan already has a chat behind it.
 *
 * Callers use this to label the entry point — "Ask about managing this" the
 * first time, "Continue chatting" once there is something to return to.
 */
export async function hasConversation(scanId: string): Promise<boolean> {
  try {
    const record = await getConversation(scanId);
    return (record?.messages.length ?? 0) > 0;
  } catch {
    return false;
  }
}

/**
 * Writes the whole thread for a scan. Called after each completed turn, so the
 * put is idempotent and the last write wins.
 *
 * On a quota failure it drops the oldest turns and retries once: losing the top
 * of a long conversation is better than losing the reply just generated.
 */
export async function saveConversation(
  scanId: string,
  messages: DiagnosisUIMessage[],
): Promise<void> {
  const write = (msgs: DiagnosisUIMessage[]) =>
    tx(
      CONVERSATIONS,
      "readwrite",
      (store) =>
        store.put({
          scan_id: scanId,
          updated_at: Date.now(),
          messages: msgs,
        } satisfies ConversationRecord) as IDBRequest<IDBValidKey>,
    );

  try {
    await write(messages);
    return;
  } catch (err) {
    if (!(err instanceof QuotaError) || messages.length <= 4) throw err;
  }
  await write(messages.slice(-4));
}

export async function deleteConversation(scanId: string): Promise<void> {
  await tx(CONVERSATIONS, "readwrite", (store) => store.delete(scanId) as IDBRequest<undefined>);
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

/**
 * Photo + leaf cutout -> one flat picture of the segmented blade.
 *
 * The cutout the API returns is an alpha mask, not a picture: opaque black
 * outside the blade, 10% black over it. Laid on the photo it reads as the leaf
 * on a black field, but on its own it is a black rectangle with a leaf-shaped
 * hole. A saved scan keeps no photo — only the overlay — so the two have to be
 * flattened together here, while the photo is still in hand, for the reopened
 * scan to have anything to show.
 *
 * Drawn at the cutout's own resolution, which is the size the API already
 * downscaled the overlay to. JPEG rather than PNG because the content is a
 * photograph: the same frame as PNG is the multi-megabyte thing the overlay had
 * to be engineered down from, and the background is flat black, which is where
 * JPEG is cheapest.
 */
export async function flattenCutout(
  photo: string,
  cutout: string,
  quality = 0.9,
): Promise<string> {
  const [base, mask] = await Promise.all([loadImage(photo), loadImage(cutout)]);

  const canvas = document.createElement("canvas");
  canvas.width = mask.naturalWidth;
  canvas.height = mask.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not composite the leaf segmentation.");

  ctx.drawImage(base, 0, 0, canvas.width, canvas.height);
  ctx.drawImage(mask, 0, 0);
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
