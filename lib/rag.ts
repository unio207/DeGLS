import { openai } from "@ai-sdk/openai";
import { embed } from "ai";

import embeddingsFile from "@/data/embeddings.json";

/**
 * Retrieval over the extension-publication corpus in `corpus/`.
 *
 * ---------------------------------------------------------------------------
 * Why an embeddings.json and not a vector database
 * ---------------------------------------------------------------------------
 * The corpus is a few dozen university-extension documents — currently well
 * under a thousand chunks. At that size a precomputed embeddings file loaded
 * into memory beats a hosted vector DB on every axis that matters here:
 *
 *   - No service to provision, no second dashboard, no bill, no API key.
 *   - No network hop on the retrieval path. A brute-force cosine scan over a
 *     few hundred 1536-d vectors is sub-millisecond; a hosted query is 30-80ms
 *     before the model has even started generating.
 *   - The index is a committed artifact, so a deploy is reproducible and the
 *     corpus is reviewable in a diff.
 *
 * The tradeoff is that this does not scale. Everything is resident in the
 * function's memory (~6 KB per chunk) and the scan is O(n). Somewhere around
 * 5-10k chunks the memory cost and cold-start parse time stop being free.
 *
 * That is why the export surface here is exactly one function, `retrieve`.
 * When the corpus outgrows this, swapping in Upstash Vector (or pgvector, or
 * anything else) means reimplementing `retrieve` and nothing else — the chat
 * route only knows about `RetrievedChunk[]`.
 *
 * Regenerate the index with `python scripts/ingest_corpus.py`.
 */

/** Provenance carried on every chunk, from the corpus file's YAML frontmatter. */
export interface CorpusSource {
  title: string;
  publisher: string;
  url?: string;
  author?: string;
  date?: string;
  crop?: string;
  disease?: string[];
  /** Originating file in `corpus/`, for debugging. */
  file: string;
}

export interface RetrievedChunk {
  id: string;
  text: string;
  /** Cosine similarity in [-1, 1]; in practice ~0.1-0.7 for this corpus. */
  score: number;
  source: CorpusSource;
}

interface StoredChunk {
  id: string;
  hash: string;
  text: string;
  chunk_index: number;
  embedding: number[];
  source: {
    title: string;
    publisher: string;
    file: string;
    url?: string;
    author?: string;
    date?: string;
    crop?: string;
    disease?: string[];
  };
}

interface EmbeddingsFile {
  model: string;
  dimensions: number;
  generated_at: string | null;
  chunk_count: number;
  chunks: StoredChunk[];
}

const EMBEDDING_MODEL = "text-embedding-3-small";

/**
 * Chunks below this cosine similarity are dropped rather than handed to the
 * model. Without a floor, an off-topic question ("what's the weather?") still
 * returns the six least-bad chunks and the model dutifully cites them. The
 * route treats an empty result as "answer from the diagnosis alone and say so".
 */
const MIN_SCORE = 0.22;

/**
 * A static import rather than a runtime `fs.readFile`: it guarantees the index
 * is in the function bundle on Vercel without any file-tracing configuration,
 * and it is parsed once at module init rather than per request.
 *
 * The tradeoff is that `data/embeddings.json` must exist at build time. It is
 * committed with `chunks: []` precisely so a fresh clone builds and the chat
 * route falls back to diagnosis-context-only answers. Deleting the file
 * outright is a loud build error, which is the right failure mode — a silently
 * empty index that never cites anything is much harder to notice.
 */
const index = embeddingsFile as unknown as EmbeddingsFile;
const chunks: StoredChunk[] = Array.isArray(index?.chunks) ? index.chunks : [];

/**
 * True when there is an ingested corpus to search. False on a fresh clone
 * before `scripts/ingest_corpus.py` has been run; the chat route uses this to
 * fall back to diagnosis-context-only answers instead of erroring.
 */
export function isCorpusReady(): boolean {
  return chunks.length > 0;
}

export function corpusStats(): { chunkCount: number; generatedAt: string | null } {
  return { chunkCount: chunks.length, generatedAt: index?.generated_at ?? null };
}

function cosineSimilarity(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Embed `query` and return the `k` most similar corpus chunks, best first.
 *
 * Returns `[]` — never throws — when the corpus is empty, the query is blank,
 * nothing clears `MIN_SCORE`, or the embedding call fails. A chat request must
 * still be answerable when retrieval is unavailable.
 *
 * This is the entire retrieval interface. Keep it that way.
 */
export async function retrieve(query: string, k = 6): Promise<RetrievedChunk[]> {
  const trimmed = query.trim();
  if (!trimmed || chunks.length === 0) return [];

  let queryEmbedding: number[];
  try {
    const result = await embed({
      model: openai.embeddingModel(EMBEDDING_MODEL),
      value: trimmed.slice(0, 8000),
    });
    queryEmbedding = result.embedding;
  } catch (error) {
    console.error("[rag] query embedding failed; answering without retrieval", error);
    return [];
  }

  const scored: RetrievedChunk[] = [];
  for (const chunk of chunks) {
    const score = cosineSimilarity(queryEmbedding, chunk.embedding);
    if (score >= MIN_SCORE) {
      scored.push({ id: chunk.id, text: chunk.text, score, source: chunk.source });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}
