# Progress Log — PaperRec (Academic Paper Repository & Recommendation System)

This document summarizes what changed between the earlier snapshot of the
codebase and the current one, based on a diff of the two ingested project
states.

## Summary

The update is focused on one theme: **automatic metadata enrichment for
papers with missing or corrupted fields**, using the same PDF-discovery
search results (Unpaywall, Crossref, Semantic Scholar, arXiv, OpenAlex)
that already power the "Find PDF Online" feature. Previously, those
sources were queried only to find a downloadable PDF and their title/
abstract/year/author data was discarded after scoring. Now that data is
reused to backfill a paper's own missing Title, Abstract, Publication
Year, and Author fields.

Supporting changes: DOI-in-text extraction straight from uploaded PDFs,
a new `enrichment_notes` audit trail column, better handling of blocked
PDF downloads, and small frontend fixes to keep the recommendation-index
"stale" banner in sync after PDF operations.

---

## Backend changes

### New: `app/services/metadata_enrichment.py`
- New module that fills in a paper's missing/corrupted **Title**,
  **Abstract**, **Publication Year**, and **Author** using candidates
  already retrieved by `pdf_finder.py`.
- Only ever *replaces* a field that is missing or looks broken — never
  overwrites a field that already looks fine.
- Title replacement uses a stricter confidence bar (`0.72`, reusing
  `MIN_CROSSREF_DOI_MATCH`) since it's the one case that overwrites an
  existing (corrupted) value; abstract/year/author use the looser
  `MIN_CONFIDENCE` bar since filling a blank field is lower risk.
- Detects corrupted titles (e.g. `"]oc.htam["`-style garbled strings)
  via `is_title_corrupted()`.
- Keywords are deliberately **not** enriched — no external source
  returns keywords in a comparable format, so merging them risked
  polluting classification/TF-IDF text.
- Returns which fields changed and writes a human-readable trace to
  `paper.enrichment_notes` (e.g. `"abstract <- semantic_scholar
  (confidence 0.76)"`) for auditability.

### `app/models/models.py`
- Added `Paper.enrichment_notes` (`Text`, nullable) — stores the audit
  trail produced by `metadata_enrichment.py`.

### `app/services/extraction.py`
- Added `_extract_doi()`: scans the first few pages of extracted PDF
  text for a DOI pattern (`10.xxxx/...`), Zotero-`recognizePDF`-style —
  read what's already on the document before making an external API
  call.
- `extract_metadata_from_pdf()` now also returns `"doi"`.
- Effect: a PDF upload can populate `Paper.doi` at insertion time
  instead of staying `None` until "Find PDF Online" is run, which lets
  `pdf_finder.py` go straight to Unpaywall (its strongest source)
  instead of first needing a fuzzy Crossref title lookup.

### `app/services/pdf_finder.py`
- `PdfCandidate` gained three enrichment-only fields: `abstract`,
  `publication_year`, `authors`. These ride along on every candidate
  (Pydantic drops unrecognized keys, so the public API response shape
  is unchanged) and are consumed by `metadata_enrichment.py`.
- Every source function (`_search_unpaywall`, `_search_semantic_scholar`,
  `_search_arxiv`, `_search_openalex`, `_search_crossref_pdf_links`) now
  populates those fields instead of discarding the data after scoring.
- `_search_semantic_scholar` now also requests the `abstract` field.
- OpenAlex abstracts are reconstructed from their inverted-index format
  via a new `_reconstruct_openalex_abstract()` helper.
- `download_and_attach_pdf()` now sends a browser-shaped `User-Agent`
  and `Accept` header — some publishers (e.g. MDPI) were returning 403
  for generic/bot-looking requests.

### `app/services/upload_paper.py`
- Added `_needs_enrichment()`: skips the enrichment network round-trip
  entirely when a paper already has all four recommendation-required
  fields and an uncorrupted title.
- Added `_try_enrich_from_pdf_discovery()` ("Step 5b"): runs the same
  PDF-discovery lookup used by "Find PDF Online" automatically on every
  upload (PDF or BibTeX), right after validation and before the first
  database commit — so a paper is inserted already enriched rather than
  inserted incomplete and patched later.
  - Conservative by design: skipped if the paper looks complete, skipped
    if there's no title to search with, and never raises (a network
    hiccup must not fail the upload).
  - Re-runs `validate_paper()` and `refresh_prepared_text()` only if
    something actually changed.

### `app/api.py`
- `GET /api/papers/{id}/find-pdf`: now also runs best-effort metadata
  enrichment using the candidates it just found, and marks the
  recommendation index stale if anything changed.
- `POST /api/papers/{id}/attach-pdf`: after attaching a downloaded PDF,
  now re-extracts metadata from that file and backfills whichever of
  Title/Abstract/Keywords/Publication Year the paper is still missing
  (never overwriting an existing value, e.g. one already set from
  BibTeX). Marks the recommendation index stale if any of those fields
  changed.

---

## Frontend changes

### `frontend/src/api.ts`
- Added `notifyRecommendationIndexStale()` — dispatches a
  `recommendation-index-stale` browser event so `AppLayout`'s listener
  can flip the "index needs updating" banner on immediately, without
  waiting for a page reload or the next status poll.

### `frontend/src/components/FindPdfPanel.tsx`
- Calls `notifyRecommendationIndexStale()` after a PDF is successfully
  attached.
- Added `isLikelyBlockedError()` and a `blockedUrl` state: when
  `attach-pdf` fails in a way that looks like the source blocked an
  automated download (non-2xx HTTP status, non-PDF response, or an
  unreachable link), the panel now surfaces a specific hint suggesting
  the user open the link manually and upload the PDF from the Upload
  page instead.

### `frontend/src/pages/repository/Upload.tsx`
- Calls `notifyRecommendationIndexStale()` after saving a paper, so the
  stale-index banner appears immediately instead of waiting for the
  next status poll.

---

## Net effect

- Papers imported from thin BibTeX citations (e.g. Google Scholar
  exports, which rarely include an abstract or DOI) are now much more
  likely to end up **valid for recommendation** without any manual
  editing, because upload-time enrichment fills in missing fields
  automatically.
- PDF uploads get a DOI "for free" when the document prints one,
  improving the accuracy and confidence of subsequent PDF-discovery
  searches.
- Every enrichment action is auditable via `Paper.enrichment_notes`.
- The UI's "recommendation index is stale" banner now reacts instantly
  to PDF-related changes instead of only on the next poll.
