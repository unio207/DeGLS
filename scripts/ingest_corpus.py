#!/usr/bin/env python3
"""
Build the DeGLS retrieval index.

    corpus/*.md (+ optional *.pdf)
      -> parse YAML frontmatter
      -> chunk with overlap, carrying metadata onto every chunk
      -> embed with OpenAI text-embedding-3-small
      -> data/embeddings.json

This is a LOCAL DEV TOOL. It never runs in production: `data/embeddings.json` is
committed and the Next.js chat route reads it directly. Its dependencies live in
scripts/requirements.txt and are deliberately kept out of api/requirements.txt,
which is size-constrained for the serverless inference function.

Usage
-----
    pip install -r scripts/requirements.txt
    python scripts/ingest_corpus.py            # incremental (default)
    python scripts/ingest_corpus.py --force    # re-embed everything
    python scripts/ingest_corpus.py --dry-run  # chunk + report, no API calls

The OpenAI key is read from the environment or from .env.local / .env at the repo
root. It is NEVER written to disk, never echoed, and never baked into the output.
A previous incarnation of this repo leaked a key into public source control; that
is precisely what this file exists to avoid.

Re-running is cheap: every chunk is keyed by a sha256 of its text plus the model
name, and vectors are reused from the existing data/embeddings.json. Editing one
corpus file only re-embeds that file's chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
OUTPUT_PATH = REPO_ROOT / "data" / "embeddings.json"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Chunk sizing is in characters, not tokens. Roughly 4 chars/token, so ~1,400
# chars lands near 350 tokens: big enough to hold a whole "Conditions favoring
# disease development" section, small enough that top-k retrieval stays precise.
CHUNK_CHARS = 1400
CHUNK_OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 120

EMBED_BATCH_SIZE = 96


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def load_api_key() -> str | None:
    """Resolve OPENAI_API_KEY from the environment, then .env.local, then .env."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()

    try:  # python-dotenv is convenient but not required
        from dotenv import load_dotenv

        for name in (".env.local", ".env"):
            path = REPO_ROOT / name
            if path.exists():
                load_dotenv(path, override=False)
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key.strip()
    except ImportError:
        pass

    # Minimal fallback parser so the script works without python-dotenv.
    for name in (".env.local", ".env"):
        path = REPO_ROOT / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "OPENAI_API_KEY":
                return v.strip().strip("'\"")
    return None


# --------------------------------------------------------------------------
# Frontmatter + parsing
# --------------------------------------------------------------------------

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Split a markdown file into (frontmatter dict, body)."""
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    block, body = match.group(1), raw[match.end() :]

    try:
        import yaml

        data = yaml.safe_load(block) or {}
        if isinstance(data, dict):
            return data, body
    except ImportError:
        pass

    # Fallback: flat `key: value` pairs, with `[a, b]` treated as a list.
    data = {}
    for line in block.splitlines():
        if ":" not in line or line.strip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip("'\"")
        if value.startswith("[") and value.endswith("]"):
            data[key.strip()] = [
                item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()
            ]
        else:
            data[key.strip()] = value
    return data, body


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        print(
            f"  ! skipping {path.name}: install pypdf (pip install -r scripts/requirements.txt)",
            file=sys.stderr,
        )
        return ""
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 - a bad PDF should not kill the run
        print(f"  ! skipping {path.name}: {exc}", file=sys.stderr)
        return ""
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(body: str) -> list[str]:
    """Paragraph-ish units. Markdown tables are kept whole so rows keep headers."""
    blocks: list[str] = []
    buffer: list[str] = []
    in_table = False

    for line in body.split("\n"):
        is_table_row = line.lstrip().startswith("|")
        if is_table_row and not in_table:
            if buffer:
                blocks.append("\n".join(buffer))
                buffer = []
            in_table = True
        elif in_table and not is_table_row and line.strip():
            blocks.append("\n".join(buffer))
            buffer = []
            in_table = False

        if not line.strip() and not in_table:
            if buffer:
                blocks.append("\n".join(buffer))
                buffer = []
            continue
        buffer.append(line)

    if buffer:
        blocks.append("\n".join(buffer))
    return [b.strip() for b in blocks if b.strip()]


def chunk_body(body: str) -> list[str]:
    """
    Greedy paragraph packing up to CHUNK_CHARS with CHUNK_OVERLAP_CHARS of trailing
    context carried into the next chunk, so a fact split across a boundary still
    appears whole in one of them.
    """
    paragraphs = split_paragraphs(normalize_whitespace(body))
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # A single oversized paragraph (long table, dense bulletin section) is
        # hard-split rather than emitted as one enormous chunk.
        if len(para) > CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                chunks.append(para[start : start + CHUNK_CHARS])
                start += CHUNK_CHARS - CHUNK_OVERLAP_CHARS
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= CHUNK_CHARS:
            current = candidate
            continue

        chunks.append(current)
        tail = current[-CHUNK_OVERLAP_CHARS:] if CHUNK_OVERLAP_CHARS else ""
        # Start the overlap at a word boundary so chunks don't open mid-word.
        if tail and " " in tail:
            tail = tail[tail.index(" ") + 1 :]
        current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if len(c.strip()) >= MIN_CHUNK_CHARS]


# --------------------------------------------------------------------------
# Documents -> chunk records
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    hash: str
    text: str
    source: dict[str, Any]
    chunk_index: int
    embedding: list[float] = field(default_factory=list)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def build_source(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    title = str(meta.get("title") or path.stem.replace("-", " ").title())
    source: dict[str, Any] = {
        "title": title,
        "publisher": str(meta.get("publisher") or "Unknown publisher"),
        "file": path.name,
    }
    for key in ("url", "author", "date", "retrieved", "crop"):
        value = meta.get(key)
        if value:
            source[key] = str(value)
    diseases = as_list(meta.get("disease"))
    if diseases:
        source["disease"] = diseases
    return source


def chunk_hash(text: str) -> str:
    return hashlib.sha256(f"{EMBEDDING_MODEL}\x00{text}".encode("utf-8")).hexdigest()


def collect_chunks() -> list[Chunk]:
    if not CORPUS_DIR.exists():
        return []

    paths = sorted(
        p
        for p in CORPUS_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".md", ".markdown", ".pdf"}
        and p.name.lower() != "readme.md"
    )

    chunks: list[Chunk] = []
    for path in paths:
        if path.suffix.lower() == ".pdf":
            body = read_pdf(path)
            meta: dict[str, Any] = {"title": path.stem.replace("-", " ").title()}
        else:
            meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))

        if not body.strip():
            continue

        source = build_source(path, meta)
        pieces = chunk_body(body)

        # Prefixing each chunk with its provenance measurably helps retrieval:
        # a query like "gray leaf spot fungicide threshold" matches a chunk whose
        # own text says "Iowa State ... threshold" but never names the disease.
        header = f"{source['title']} ({source['publisher']})"
        for i, piece in enumerate(pieces):
            text = f"{header}\n\n{piece}"
            chunks.append(
                Chunk(
                    id=f"{path.stem}::{i}",
                    hash=chunk_hash(text),
                    text=text,
                    source=source,
                    chunk_index=i,
                )
            )
        print(f"  {path.name}: {len(pieces)} chunks")

    return chunks


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


def load_existing_vectors() -> dict[str, list[float]]:
    """hash -> vector, from a previous run. This is the whole cache."""
    if not OUTPUT_PATH.exists():
        return {}
    try:
        data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if data.get("model") != EMBEDDING_MODEL:
        return {}
    cache: dict[str, list[float]] = {}
    for chunk in data.get("chunks", []):
        h, vec = chunk.get("hash"), chunk.get("embedding")
        if h and isinstance(vec, list) and vec:
            cache[h] = vec
    return cache


def batched(items: list[Chunk], size: int) -> Iterable[list[Chunk]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_missing(client: Any, pending: list[Chunk]) -> None:
    total = len(pending)
    done = 0
    for batch in batched(pending, EMBED_BATCH_SIZE):
        for attempt in range(5):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=[c.text for c in batch],
                )
                break
            except Exception as exc:  # noqa: BLE001 - retry transient API failures
                if attempt == 4:
                    raise
                wait = 2**attempt
                print(f"  ! embed failed ({exc}); retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
        for chunk, item in zip(batch, response.data):
            chunk.embedding = item.embedding
        done += len(batch)
        print(f"  embedded {done}/{total}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-embed every chunk")
    parser.add_argument(
        "--dry-run", action="store_true", help="chunk and report without calling the API"
    )
    args = parser.parse_args()

    print(f"Reading corpus from {CORPUS_DIR}")
    chunks = collect_chunks()
    if not chunks:
        print(
            "No corpus documents found. The chat route degrades to "
            "diagnosis-context-only answers, so this is survivable — but add "
            "markdown files to corpus/ to get citations.",
        )
        return 0

    print(f"\n{len(chunks)} chunks from the corpus.")

    if args.dry_run:
        chars = sum(len(c.text) for c in chunks)
        print(f"Dry run: {chars:,} characters, ~{chars // 4:,} tokens. No API calls made.")
        return 0

    cache = {} if args.force else load_existing_vectors()
    for chunk in chunks:
        cached = cache.get(chunk.hash)
        if cached:
            chunk.embedding = cached

    pending = [c for c in chunks if not c.embedding]
    print(f"{len(chunks) - len(pending)} reused from cache, {len(pending)} to embed.")

    if pending:
        api_key = load_api_key()
        if not api_key:
            print(
                "\nOPENAI_API_KEY not found.\n"
                "Set it in the environment or in .env.local at the repo root:\n"
                "    OPENAI_API_KEY=sk-...\n"
                "Never commit that file.",
                file=sys.stderr,
            )
            return 1
        try:
            from openai import OpenAI
        except ImportError:
            print(
                "openai package missing. pip install -r scripts/requirements.txt",
                file=sys.stderr,
            )
            return 1

        embed_missing(OpenAI(api_key=api_key), pending)

    payload = {
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunk_count": len(chunks),
        "chunks": [
            {
                "id": c.id,
                "hash": c.hash,
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "embedding": [round(v, 6) for v in c.embedding],
            }
            for c in chunks
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({len(chunks)} chunks, {size_mb:.1f} MB)")
    print("Commit it - the app reads this file directly and never runs this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
