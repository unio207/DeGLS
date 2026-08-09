# DeGLS corpus

Source material for the DeGLS management assistant. Everything here is openly published
university extension / Crop Protection Network material on the three corn foliar diseases DeGLS
classifies: **gray leaf spot**, **northern corn leaf blight**, and **common rust**.

`scripts/ingest_corpus.py` reads every `*.md` (and `*.pdf`, if any) in this directory, chunks it
with overlap, embeds each chunk with OpenAI `text-embedding-3-small`, and writes
`data/embeddings.json`. `lib/rag.ts` loads that file and does an in-memory cosine similarity
search at request time. Every chunk keeps its source metadata, so the chat route can cite the
publication and link to the original URL.

## Frontmatter contract

Every file must start with YAML frontmatter. The ingester copies these fields onto each chunk;
`title` and `url` are what the UI renders as a citation.

```yaml
---
title: "Gray Leaf Spot of Corn"          # required — shown as the citation label
url: "https://example.org/page"          # required — the tappable citation link
publisher: "Crop Protection Network"     # required — shown after the title
author: "Kiersten Wise"                  # optional
date: "2023-04-20"                       # optional
crop: "corn"                             # optional
disease: ["gray leaf spot"]              # optional, string or list
---
```

Anything after the frontmatter is treated as body text.

## What is here

| File | Publisher | Diseases |
|---|---|---|
| `cpn-gray-leaf-spot-encyclopedia.md` | Crop Protection Network | GLS |
| `cpn-northern-corn-leaf-blight-encyclopedia.md` | Crop Protection Network | NCLB |
| `cpn-common-rust-encyclopedia.md` | Crop Protection Network | Common rust |
| `cpn-overview-northern-corn-leaf-blight.md` | Crop Protection Network | NCLB |
| `cpn-fungicide-efficacy-corn-foliar-diseases.md` | Crop Protection Network (CDWG) | All three + lookalikes |
| `cpn-bacterial-leaf-streak-vs-gray-leaf-spot.md` | Crop Protection Network | GLS vs BLS |
| `purdue-bp-56-w-gray-leaf-spot.md` | Purdue Extension | GLS |
| `purdue-bp-84-w-northern-corn-leaf-blight.md` | Purdue Extension | NCLB |
| `unl-cropwatch-gray-leaf-spot.md` | Nebraska Extension CropWatch | GLS |
| `unl-cropwatch-common-rust.md` | Nebraska Extension CropWatch | Common rust |
| `unl-cropwatch-differentiating-corn-leaf-diseases.md` | Nebraska Extension CropWatch | Field ID / lookalikes |
| `isu-before-applying-fungicides-stop-look-consider.md` | Iowa State Extension | Fungicide decision-making |
| `isu-update-corn-diseases-fungicide-decisions.md` | Iowa State Extension | Infection conditions, timing |
| `isu-fungicide-decisions-2026.md` | Iowa State Extension | ROI, hybrid susceptibility, planting date |
| `cpn-southern-rust-encyclopedia.md` | Crop Protection Network | Southern rust vs common rust |
| `illinois-corn-diseases-to-scout-july.md` | University of Illinois Extension | GLS, NCLB, tar spot |
| `wisconsin-early-season-disease-update.md` | UW-Madison Extension | Timing, ROI tools |

### Abiotic look-alikes — symptoms that are NOT disease

The classifier and the severity mask can both read uniform yellowing as disease. These files
exist so the assistant can talk a grower through chlorosis-vs-lesion themselves.

| File | Publisher | Covers |
|---|---|---|
| `isu-yellow-corn-plants.md` | Iowa State Extension | Seven non-disease causes of yellow corn |
| `isu-nitrogen-vs-sulfur-deficiency.md` | Iowa State Extension | N (lower leaves, V down midrib) vs S (new leaves) |
| `isu-potassium-deficiency-corn.md` | Iowa State Extension | Lower-leaf margin firing vs N's midrib V |
| `purdue-corn-responses-to-drought-stress.md` | Purdue Agronomy | Leaf rolling, green→gray→straw firing |
| `purdue-top-leaf-death-and-senescence.md` | Purdue Agronomy | Natural senescence; uniform-vs-random plant pattern |
| `psu-early-season-herbicide-injury-corn.md` | Penn State Extension | Bleaching, interveinal chlorosis by herbicide group |

The two Purdue bulletins and the CPN fungicide efficacy table were transcribed from the source
PDFs; the rest were transcribed from the live web pages. Each file's frontmatter points at the
canonical URL — **check the original before acting on anything**, especially the fungicide
efficacy table, which CPN revises annually.

Files carrying a `retrieved:` frontmatter field were fetched from the live page on that date.
Files without it predate the convention; their `date:` is the publication date, not a fetch
date.

## Status: the index has not been built yet

`data/embeddings.json` is currently committed as a valid but **empty** index
(`"chunks": []`). Building it for real needs an `OPENAI_API_KEY`, which was not
available when the corpus was written. Until you run the ingester the chat route
answers from the diagnosis context alone and says it has no sources — it does not
error. Run the two commands below and the citations turn on.

## Adding your own material

1. Drop a new `.md` file in this directory with the frontmatter above. Name it
   `<publisher>-<topic>.md`. Plain PDFs also work if `pypdf` is installed — the ingester will
   pull their text, but a PDF has no frontmatter, so its citation falls back to the filename.
   Transcribing a PDF into markdown with real frontmatter gives much better citations.
2. Re-run the ingester:

   ```bash
   pip install -r scripts/requirements.txt
   # OPENAI_API_KEY from the environment or .env.local
   python scripts/ingest_corpus.py
   ```

   It is content-hash cached, so re-running after editing one file only re-embeds that file's
   chunks. Nothing else costs a token.
3. Commit the regenerated `data/embeddings.json` — the deployed app reads it directly and never
   runs the ingester.

Only add material you are allowed to redistribute. Everything currently here is public extension
outreach published for grower use, and every chunk carries attribution back to its publisher.
