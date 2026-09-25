"""
Finds a candidate PDF for a paper that doesn't have one stored yet --
e.g. one just imported from a Google Scholar / pasted BibTeX citation.

Search strategy (in order, first successful source wins but all found
candidates are returned so the user can pick):

    0. If the paper has no DOI on file (common for Google Scholar
       BibTeX exports, which almost never include one), Crossref is
       used to resolve one from the title/author, so Unpaywall --
       the strongest source -- isn't skipped just because the .bib
       file didn't have a doi field. (PDF uploads increasingly skip
       this step entirely: app/services/extraction.py now reads a DOI
       straight off the PDF's own text first, Zotero-recognizePDF
       style, so there's often already a DOI on file by the time this
       runs.)
    1. Unpaywall, by DOI (if the paper has one, or one was just
       resolved via Crossref) -- the most reliable source for a
       legally open-access PDF.
    2. Crossref's own `link` metadata, by DOI -- some publishers list
       a direct PDF link in Crossref itself, independent of Unpaywall.
    3. Semantic Scholar, by title -- covers papers without a DOI on
       file, and also surfaces a DOI we didn't have yet.
    4. arXiv, by title -- useful for preprints Unpaywall/Semantic
       Scholar don't have OA links for. Falls back to a looser,
       word-based query if the exact-phrase query returns nothing.
    5. OpenAlex, by title -- aggregates OA locations from Crossref,
       PubMed, and institutional repositories.
    6. (Optional) Google Scholar itself, via SerpApi -- Scholar has no
       public API and scraping it directly gets blocked/CAPTCHA'd and
       breaks its ToS, so this is only enabled if a SERPAPI_API_KEY
       environment variable is set. It's a no-op otherwise.

Candidates are scored on title similarity, then adjusted using two
extra signals when they're available on both sides:

    - Publication year   -- exact match boosts confidence, a gap of
      more than one year penalizes it (two unrelated papers rarely
      share both a near-identical title AND the same year).
    - Author overlap      -- if any of the paper's BibTeX author last
      names appears among the candidate's listed authors, that's a
      strong positive signal; if both sides have author data and none
      overlap at all, that's a negative signal.

Title similarity itself blends two measures (see _title_similarity):
a character-sequence ratio and a word-token overlap ratio, taking the
higher of the two. A pure character-sequence ratio is brutally
sensitive to word reordering and subtitle differences ("X: a study of
Y" vs "X -- a study of Y"), which was silently killing otherwise-good
matches.

Metadata enrichment (see app/services/metadata_enrichment.py):
    Every search function below already downloads a candidate's title,
    abstract, publication year, and author list from the external
    source -- it used to use that data only to compute the confidence
    score above, then discard everything except the DOI. PdfCandidate
    now carries abstract/publication_year/authors alongside the fields
    the API response uses, so metadata_enrichment.py can fill a
    paper's own missing/corrupted fields from whichever candidate
    matched best, instead of that data being thrown away.

Once the user confirms a candidate, download_and_attach_pdf() downloads
it and -- before saving it -- runs two checks directly against the
downloaded file itself, rather than trusting a search API's metadata:

    1. Title re-extraction -- the same extractor used for normal PDF
       uploads re-reads the PDF's own title and compares it to the
       paper's BibTeX title.
    2. DOI-in-text check -- if the paper has a DOI on file, the PDF's
       text (first few pages) is scanned for that exact DOI string.
       A DOI printed on the document itself is very strong evidence
       it's the right paper -- stronger than any title-similarity
       heuristic, since it isn't fooled by similar-sounding papers.

Every source here only surfaces legally open-access copies (publisher
OA, repository copies, or preprints), never a scraped/paywalled PDF.
Nothing is downloaded until the user confirms a candidate from
GET /api/papers/{id}/find-pdf via POST /api/papers/{id}/attach-pdf.

DEBUGGING "it returns nothing":
    Set the logger for this module to DEBUG
    (logging.getLogger("app.services.recommendation.pdf_finder")) and
    call find_pdf_candidates(paper, include_rejected=True). Every
    source now logs why it came back empty (network error, no OA
    location, rate-limited, etc.) instead of silently returning [],
    and rejected candidates are returned with their score and the
    reason they were filtered out.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, asdict, field
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import NamedTemporaryFile

import pdfplumber
import requests

from app.models.models import Paper
from app.services.storage import save_paper_file
from app.services.extraction import extract_metadata_from_pdf

logger = logging.getLogger(__name__)

# Contact address required by Unpaywall's terms of use ("polite pool").
# Swap this for a real contact address for this deployment.
UNPAYWALL_CONTACT_EMAIL = "paperrec-dev@example.com"

# Optional: set these environment variables to unlock higher rate
# limits / extra sources. Both are no-ops if left unset.
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

REQUEST_TIMEOUT = 10  # seconds
MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB safety cap
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5

# Below this title-similarity score, a candidate is dropped rather than
# shown -- a low-confidence "match" does more harm than good in a
# confirm-first flow, since the person is trusting the title at a glance.
# Lowered from 0.55: the blended similarity score in _title_similarity
# is stricter about *wrong* matches than plain SequenceMatcher was, so
# this can afford to be a bit more permissive without letting garbage
# through.
MIN_CONFIDENCE = 0.5

# Below this title-similarity score between the paper's BibTeX title
# and the title actually extracted from the downloaded PDF, the PDF is
# rejected as a likely wrong-paper match rather than attached.
MIN_DOWNLOAD_TITLE_MATCH = 0.4

# How many pages of the downloaded PDF to scan when checking whether
# its text contains the paper's known DOI.
DOI_SCAN_MAX_PAGES = 3

# Minimum combined (title + author) score required before a
# Crossref-resolved DOI is trusted enough to feed into Unpaywall.
# Lowered from 0.8 -- that threshold was rejecting a lot of true
# matches whose Scholar-exported title differs slightly from the
# publisher's title and which have no author data to cross-check
# against. Still well above MIN_CONFIDENCE since a wrong DOI here
# would feed a wrong-paper PDF straight into Unpaywall as if
# confirmed.
MIN_CROSSREF_DOI_MATCH = 0.72


@dataclass
class PdfCandidate:
    url: str
    source: str            # "unpaywall" | "crossref" | "semantic_scholar" | "arxiv" | "openalex" | "google_scholar"
    title: str | None
    confidence: float      # 0.0-1.0, title + year + author based
    landing_page_url: str | None = None
    license: str | None = None

    # -------------------------------------------------------------
    # Enrichment-only fields.
    #
    # Not part of PdfCandidateOut (app/schemas.py) -- FastAPI/Pydantic
    # silently drops unrecognized keys, so these ride along on every
    # candidate returned by find_pdf_candidates() without changing the
    # /find-pdf API response shape. metadata_enrichment.py reads them
    # to fill in a paper's own missing/corrupted title, abstract,
    # publication year, or author -- data every source below already
    # fetches to compute `confidence`, which previously got thrown
    # away right after scoring.
    # -------------------------------------------------------------
    abstract: str | None = None
    publication_year: int | None = None
    authors: list[str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RejectedCandidate:
    url: str
    source: str
    title: str | None
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------
# HTTP helper with basic retry/backoff (mainly for Semantic Scholar's
# aggressive unauthenticated rate limiting, but used generally so any
# source's transient failure gets one or two retries instead of
# silently giving up)
# ---------------------------------------------------------------------

def _get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    source: str,
) -> requests.Response | None:
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 2):  # e.g. 3 total attempts
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            last_error = f"network error: {exc}"
            logger.debug("[%s] attempt %d failed: %s", source, attempt, last_error)
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        if response.status_code == 429 and attempt <= MAX_RETRIES:
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else RETRY_BACKOFF_SECONDS * attempt
            logger.debug(
                "[%s] rate-limited (429), retrying in %.1fs (attempt %d)",
                source, delay, attempt,
            )
            time.sleep(delay)
            continue

        if not response.ok:
            logger.debug(
                "[%s] HTTP %d for %s", source, response.status_code, url
            )
            return response  # let the caller decide; still return it

        return response

    logger.debug("[%s] giving up after retries: %s", source, last_error)
    return None


# ---------------------------------------------------------------------
# Title-similarity helper (used to rank/filter candidates)
# ---------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "for", "on", "in", "to",
    "with", "using", "toward", "towards", "via", "through",
}


def _normalize_title(title: str | None) -> str:
    if not title:
        return ""
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    return " ".join(title.split())


def _title_tokens(title: str | None) -> set[str]:
    normalized = _normalize_title(title)
    return {t for t in normalized.split() if len(t) > 2 and t not in _STOPWORDS}


def _token_overlap_score(a: str | None, b: str | None) -> float:
    """
    Jaccard overlap of significant word tokens. Robust to word
    reordering and short subtitle differences that punish a pure
    character-sequence comparison.
    """
    tokens_a, tokens_b = _title_tokens(a), _title_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _title_similarity(a: str | None, b: str | None) -> float:
    """
    Blends a character-sequence ratio with a word-token overlap ratio
    and takes the higher of the two. Either measure alone has a
    failure mode the other doesn't: SequenceMatcher tanks on reordered
    words or a dropped/added subtitle; token overlap tanks on
    near-duplicate phrasing with very different wording. A genuine
    match usually scores well on at least one of them.
    """
    a_norm, b_norm = _normalize_title(a), _normalize_title(b)
    if not a_norm or not b_norm:
        return 0.0

    sequence_score = SequenceMatcher(None, a_norm, b_norm).ratio()
    token_score = _token_overlap_score(a, b)

    return max(sequence_score, token_score)


# ---------------------------------------------------------------------
# Year cross-check helper (used to boost/penalize title-based scores)
# ---------------------------------------------------------------------

def _apply_year_adjustment(
    base_score: float,
    paper_year: int | None,
    candidate_year: int | None,
) -> float:
    """
    Nudges a confidence score using publication year, when both years
    are known:

        - Exact year match      -> small confidence boost.
        - Off by one year       -> left unchanged (common due to
                                    preprint vs. publication-date gaps).
        - Off by more than one  -> confidence penalty. Two unrelated
                                    papers rarely share both a very
                                    similar title AND the same year, so
                                    a year mismatch is a signal this is
                                    the wrong paper.
    """
    if paper_year is None or candidate_year is None:
        return base_score

    diff = abs(paper_year - candidate_year)

    if diff == 0:
        return min(1.0, base_score + 0.1)
    if diff <= 1:
        return base_score
    return max(0.0, base_score - 0.2)


# ---------------------------------------------------------------------
# Author cross-check helper (used to boost/penalize title-based scores)
# ---------------------------------------------------------------------

def _extract_author_last_names(author_field: str | None) -> list[str]:
    """
    Pulls last names out of the paper's stored author string.

    The repository stores authors in BibTeX-derived form, e.g.:

        "Vaswani, Ashish and Shazeer, Noam and Parmar, Niki"

    or occasionally just a plain name without a comma. Either way, the
    piece before the first comma (or the whole chunk, if no comma) is
    treated as the last name for matching purposes.
    """
    if not author_field:
        return []

    last_names = []

    for chunk in author_field.split(" and "):
        chunk = chunk.strip()
        if not chunk:
            continue

        last_name = chunk.split(",")[0].strip()
        if len(last_name) >= 2:
            last_names.append(last_name.lower())

    return last_names


def _apply_author_adjustment(
    base_score: float,
    paper_last_names: list[str],
    candidate_authors: list[str] | None,
) -> float:
    """
    Nudges a confidence score using author overlap, when both sides
    have author data:

        - At least one shared author -> small confidence boost.
        - Both sides known, zero overlap -> confidence penalty.
        - Either side missing author data -> left unchanged (can't
          compare what isn't there).
    """
    if not paper_last_names or not candidate_authors:
        return base_score

    candidate_text = " ".join(candidate_authors).lower()

    has_overlap = any(
        last_name in candidate_text for last_name in paper_last_names
    )

    if has_overlap:
        return min(1.0, base_score + 0.1)

    return max(0.0, base_score - 0.1)


# ---------------------------------------------------------------------
# OpenAlex abstract reconstruction
# ---------------------------------------------------------------------

def _reconstruct_openalex_abstract(
    inverted_index: dict[str, list[int]] | None,
) -> str | None:
    """
    OpenAlex returns an abstract as `abstract_inverted_index` -- a
    word -> [positions] map instead of running text (a copyright-
    avoidance format) -- e.g.:

        {"Neural": [0], "networks": [1], "are": [2], ...}

    This rebuilds a plain-text abstract by placing each word back at
    its recorded position(s). Good enough for keyword search / display
    purposes; not guaranteed to reproduce the original's exact
    formatting or punctuation spacing.
    """
    if not inverted_index:
        return None

    try:
        max_position = max(
            position
            for positions in inverted_index.values()
            for position in positions
        )
    except ValueError:
        return None

    words: list[str] = [""] * (max_position + 1)

    for word, positions in inverted_index.items():
        for position in positions:
            if 0 <= position <= max_position:
                words[position] = word

    abstract = " ".join(word for word in words if word)
    abstract = " ".join(abstract.split())

    return abstract or None


# ---------------------------------------------------------------------
# DOI resolution via Crossref (for papers with no DOI on file --
# common for Google Scholar BibTeX exports, and for any PDF upload
# where extraction.py's own DOI-in-text scan came up empty)
# ---------------------------------------------------------------------

def _crossref_search(paper: Paper, rows: int = 5) -> list[dict]:
    if not paper.title:
        return []

    response = _get_with_retry(
        "https://api.crossref.org/works",
        params={
            "query.bibliographic": paper.title,
            "rows": rows,
            "mailto": UNPAYWALL_CONTACT_EMAIL,
        },
        source="crossref",
    )

    if response is None:
        logger.debug("[crossref] no response after retries for title=%r", paper.title)
        return []

    if not response.ok:
        logger.debug(
            "[crossref] HTTP %d for title=%r", response.status_code, paper.title
        )
        return []

    return response.json().get("message", {}).get("items", []) or []


def _resolve_doi_via_crossref(paper: Paper) -> str | None:
    """
    Looks up a likely DOI for this paper using Crossref's free
    bibliographic search, when the paper has no DOI on file. Only
    returns a DOI if the top result's title (and author overlap, when
    available) is a strong match -- otherwise returns None rather
    than guessing.
    """
    items = _crossref_search(paper)
    paper_last_names = _extract_author_last_names(paper.author)

    best_score = 0.0
    best_doi = None

    for item in items:
        candidate_title = " ".join(item.get("title") or [])
        score = _title_similarity(paper.title, candidate_title)

        candidate_authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in (item.get("author") or [])
            if a.get("family")
        ]
        score = _apply_author_adjustment(score, paper_last_names, candidate_authors)

        if score > best_score:
            best_score = score
            best_doi = item.get("DOI")

    if best_doi and best_score >= MIN_CROSSREF_DOI_MATCH:
        logger.debug(
            "[crossref] resolved DOI %s for title=%r (score=%.2f)",
            best_doi, paper.title, best_score,
        )
        return best_doi

    logger.debug(
        "[crossref] no confident DOI match for title=%r (best_score=%.2f, "
        "threshold=%.2f)",
        paper.title, best_score, MIN_CROSSREF_DOI_MATCH,
    )
    return None


def _search_crossref_pdf_links(paper: Paper) -> list[PdfCandidate]:
    """
    Crossref sometimes lists a direct `link` entry with
    content-type "application/pdf" for a work -- independent of
    whether Unpaywall has indexed an OA copy. Cheap extra source once
    we're already querying Crossref for DOI resolution.
    """
    items = _crossref_search(paper)
    paper_last_names = _extract_author_last_names(paper.author)

    candidates: list[PdfCandidate] = []

    for item in items:
        pdf_link = next(
            (
                link.get("URL")
                for link in (item.get("link") or [])
                if (link.get("content-type") or "").lower() == "application/pdf"
            ),
            None,
        )
        if not pdf_link:
            continue

        candidate_title = " ".join(item.get("title") or [])
        candidate_year = None
        for date_field in ("published-print", "published-online", "issued"):
            parts = (item.get(date_field) or {}).get("date-parts")
            if parts and parts[0]:
                candidate_year = parts[0][0]
                break

        candidate_authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in (item.get("author") or [])
            if a.get("family")
        ]

        base_confidence = _title_similarity(paper.title, candidate_title)
        confidence = _apply_year_adjustment(
            base_confidence, paper.publication_year, candidate_year
        )
        confidence = _apply_author_adjustment(
            confidence, paper_last_names, candidate_authors
        )

        candidates.append(
            PdfCandidate(
                url=pdf_link,
                source="crossref",
                title=candidate_title or None,
                confidence=confidence,
                landing_page_url=item.get("URL"),
                publication_year=candidate_year,
                authors=candidate_authors or None,
            )
        )

    return candidates


# ---------------------------------------------------------------------
# Source 1 -- Unpaywall (by DOI)
# ---------------------------------------------------------------------

def _search_unpaywall(paper: Paper) -> list[PdfCandidate]:
    if not paper.doi:
        logger.debug("[unpaywall] skipped: paper has no DOI")
        return []

    doi = paper.doi.strip()
    url = f"https://api.unpaywall.org/v2/{doi}"

    response = _get_with_retry(
        url, params={"email": UNPAYWALL_CONTACT_EMAIL}, source="unpaywall"
    )

    if response is None:
        logger.debug("[unpaywall] no response after retries for DOI=%s", doi)
        return []

    if not response.ok:
        logger.debug("[unpaywall] HTTP %d for DOI=%s", response.status_code, doi)
        return []

    data = response.json()
    candidates: list[PdfCandidate] = []

    locations = []
    if data.get("best_oa_location"):
        locations.append(data["best_oa_location"])
    for loc in data.get("oa_locations", []) or []:
        if loc not in locations:
            locations.append(loc)

    if not locations:
        logger.debug("[unpaywall] DOI=%s found but has no OA location", doi)

    result_title = data.get("title") or paper.title
    result_year = data.get("year")

    result_authors = [
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in (data.get("z_authors") or [])
        if a.get("family")
    ]

    paper_last_names = _extract_author_last_names(paper.author)

    for loc in locations:
        pdf_url = loc.get("url_for_pdf") or loc.get("url")
        if not pdf_url:
            continue

        base_confidence = max(_title_similarity(paper.title, result_title), 0.9)
        confidence = _apply_year_adjustment(
            base_confidence, paper.publication_year, result_year
        )
        confidence = _apply_author_adjustment(
            confidence, paper_last_names, result_authors
        )

        candidates.append(
            PdfCandidate(
                url=pdf_url,
                source="unpaywall",
                title=result_title,
                confidence=confidence,
                landing_page_url=loc.get("url_for_landing_page"),
                license=loc.get("license"),
                publication_year=result_year,
                authors=result_authors or None,
                # Unpaywall's API doesn't return an abstract.
            )
        )

    return candidates


# ---------------------------------------------------------------------
# Source 2 -- Semantic Scholar (by title)
# ---------------------------------------------------------------------

def _search_semantic_scholar(paper: Paper) -> list[PdfCandidate]:
    if not paper.title:
        return []

    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else None

    response = _get_with_retry(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": paper.title,
            "limit": 5,
            # "abstract" added so metadata_enrichment.py can offer it
            # as a fill-in when a paper's own abstract extraction
            # failed or came back too short to pass validate_paper().
            "fields": "title,year,authors,abstract,openAccessPdf,externalIds",
        },
        headers=headers,
        source="semantic_scholar",
    )

    if response is None:
        logger.debug(
            "[semantic_scholar] no response after retries for title=%r", paper.title
        )
        return []

    if not response.ok:
        if response.status_code == 429:
            logger.debug(
                "[semantic_scholar] still rate-limited after retries -- consider "
                "setting SEMANTIC_SCHOLAR_API_KEY for a higher quota"
            )
        else:
            logger.debug(
                "[semantic_scholar] HTTP %d for title=%r",
                response.status_code, paper.title,
            )
        return []

    candidates: list[PdfCandidate] = []
    paper_last_names = _extract_author_last_names(paper.author)

    results = response.json().get("data", []) or []
    if not results:
        logger.debug("[semantic_scholar] zero results for title=%r", paper.title)

    for result in results:
        pdf_info = result.get("openAccessPdf")
        if not pdf_info or not pdf_info.get("url"):
            continue

        result_title = result.get("title")
        result_year = result.get("year")
        result_authors = [
            a.get("name", "")
            for a in (result.get("authors") or [])
            if a.get("name")
        ]

        base_confidence = _title_similarity(paper.title, result_title)
        confidence = _apply_year_adjustment(
            base_confidence, paper.publication_year, result_year
        )
        confidence = _apply_author_adjustment(
            confidence, paper_last_names, result_authors
        )

        candidates.append(
            PdfCandidate(
                url=pdf_info["url"],
                source="semantic_scholar",
                title=result_title,
                confidence=confidence,
                license=pdf_info.get("license"),
                abstract=result.get("abstract"),
                publication_year=result_year,
                authors=result_authors or None,
            )
        )

    return candidates


# ---------------------------------------------------------------------
# Source 3 -- arXiv (by title)
# ---------------------------------------------------------------------

def _parse_arxiv_entries(xml_text: str) -> list[re.Match]:
    return re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)


def _run_arxiv_query(query: str) -> list[str]:
    response = _get_with_retry(
        "http://export.arxiv.org/api/query",
        params={"search_query": query, "start": 0, "max_results": 5},
        source="arxiv",
    )

    if response is None or not response.ok:
        logger.debug("[arxiv] query failed: %r", query)
        return []

    return _parse_arxiv_entries(response.text)


def _search_arxiv(paper: Paper) -> list[PdfCandidate]:
    if not paper.title:
        return []

    safe_title = re.sub(r'["\\]', "", paper.title).strip()

    entries = _run_arxiv_query(f'ti:"{safe_title}"')

    if not entries:
        tokens = sorted(_title_tokens(paper.title))[:8]
        if tokens:
            loose_query = " AND ".join(f'ti:{token}' for token in tokens)
            logger.debug(
                "[arxiv] exact-phrase query returned nothing, retrying "
                "with loose query: %s", loose_query,
            )
            entries = _run_arxiv_query(loose_query)

    if not entries:
        logger.debug("[arxiv] zero results for title=%r", paper.title)
        return []

    candidates: list[PdfCandidate] = []
    paper_last_names = _extract_author_last_names(paper.author)

    for entry in entries:
        title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
        id_match = re.search(r"<id>(.*?)</id>", entry, re.DOTALL)
        published_match = re.search(r"<published>(\d{4})", entry)
        summary_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
        author_matches = re.findall(
            r"<author>\s*<name>(.*?)</name>", entry, re.DOTALL
        )

        if not id_match:
            continue

        entry_title = (
            " ".join(title_match.group(1).split()) if title_match else None
        )

        entry_year = (
            int(published_match.group(1)) if published_match else None
        )

        entry_abstract = (
            " ".join(summary_match.group(1).split())
            if summary_match
            else None
        )

        entry_authors = [name.strip() for name in author_matches if name.strip()]

        abs_url = id_match.group(1).strip()
        arxiv_id_match = re.search(r"abs/(.+)$", abs_url)
        if not arxiv_id_match:
            continue

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id_match.group(1)}.pdf"

        base_confidence = _title_similarity(paper.title, entry_title)
        confidence = _apply_year_adjustment(
            base_confidence, paper.publication_year, entry_year
        )
        confidence = _apply_author_adjustment(
            confidence, paper_last_names, entry_authors
        )

        candidates.append(
            PdfCandidate(
                url=pdf_url,
                source="arxiv",
                title=entry_title,
                confidence=confidence,
                landing_page_url=abs_url,
                abstract=entry_abstract,
                publication_year=entry_year,
                authors=entry_authors or None,
            )
        )

    return candidates


# ---------------------------------------------------------------------
# Source 4 -- OpenAlex (by title)
# ---------------------------------------------------------------------

def _search_openalex(paper: Paper) -> list[PdfCandidate]:
    if not paper.title:
        return []

    response = _get_with_retry(
        "https://api.openalex.org/works",
        params={
            "search": paper.title,
            "per-page": 5,
            "mailto": UNPAYWALL_CONTACT_EMAIL,
        },
        source="openalex",
    )

    if response is None:
        logger.debug("[openalex] no response after retries for title=%r", paper.title)
        return []

    if not response.ok:
        logger.debug(
            "[openalex] HTTP %d for title=%r", response.status_code, paper.title
        )
        return []

    candidates: list[PdfCandidate] = []
    paper_last_names = _extract_author_last_names(paper.author)

    results = response.json().get("results", []) or []
    if not results:
        logger.debug("[openalex] zero results for title=%r", paper.title)

    for result in results:
        oa_location = result.get("best_oa_location") or {}
        pdf_url = oa_location.get("pdf_url")

        if not pdf_url:
            continue

        result_title = result.get("title")
        result_year = result.get("publication_year")

        result_authors = [
            (authorship.get("author") or {}).get("display_name", "")
            for authorship in (result.get("authorships") or [])
            if (authorship.get("author") or {}).get("display_name")
        ]

        # OpenAlex returns the abstract as a word -> position map
        # rather than running text (a copyright-avoidance format);
        # rebuild it into plain text for enrichment purposes.
        result_abstract = _reconstruct_openalex_abstract(
            result.get("abstract_inverted_index")
        )

        base_confidence = _title_similarity(paper.title, result_title)
        confidence = _apply_year_adjustment(
            base_confidence, paper.publication_year, result_year
        )
        confidence = _apply_author_adjustment(
            confidence, paper_last_names, result_authors
        )

        candidates.append(
            PdfCandidate(
                url=pdf_url,
                source="openalex",
                title=result_title,
                confidence=confidence,
                landing_page_url=oa_location.get("landing_page_url"),
                license=oa_location.get("license"),
                abstract=result_abstract,
                publication_year=result_year,
                authors=result_authors or None,
            )
        )

    return candidates


# ---------------------------------------------------------------------
# Source 5 (optional) -- Google Scholar via SerpApi
# ---------------------------------------------------------------------
#
# Google Scholar has no public API. Scraping scholar.google.com
# directly is unreliable (CAPTCHAs, IP blocks) and against Google's
# ToS, so this codebase does not do that. SerpApi is a third-party,
# ToS-compliant paid service that proxies Google Scholar search
# results and includes any direct PDF link Scholar shows next to a
# result. This function is a complete no-op unless a SERPAPI_API_KEY
# environment variable is set, so it costs nothing to leave in place.

def _search_google_scholar_via_serpapi(paper: Paper) -> list[PdfCandidate]:
    if not SERPAPI_API_KEY or not paper.title:
        return []

    response = _get_with_retry(
        "https://serpapi.com/search",
        params={
            "engine": "google_scholar",
            "q": paper.title,
            "api_key": SERPAPI_API_KEY,
        },
        source="google_scholar",
    )

    if response is None or not response.ok:
        logger.debug(
            "[google_scholar] SerpApi request failed for title=%r", paper.title
        )
        return []

    candidates: list[PdfCandidate] = []
    paper_last_names = _extract_author_last_names(paper.author)

    for result in response.json().get("organic_results", []) or []:
        pdf_resource = next(
            (
                res.get("link")
                for res in (result.get("resources") or [])
                if (res.get("file_format") or "").upper() == "PDF"
            ),
            None,
        )
        if not pdf_resource:
            continue

        result_title = (result.get("title") or "").strip()

        publication_info = result.get("publication_info", {}) or {}
        result_authors = [
            author.get("name", "")
            for author in (publication_info.get("authors") or [])
            if author.get("name")
        ]

        base_confidence = _title_similarity(paper.title, result_title)
        confidence = _apply_author_adjustment(
            base_confidence, paper_last_names, result_authors
        )

        candidates.append(
            PdfCandidate(
                url=pdf_resource,
                source="google_scholar",
                title=result_title or None,
                confidence=confidence,
                landing_page_url=result.get("link"),
                # SerpApi's organic_results snippet is the closest
                # thing to an abstract Scholar exposes -- close enough
                # in spirit but often truncated with an ellipsis, so
                # it's deliberately left out of enrichment rather than
                # risking a half-sentence abstract.
                authors=result_authors or None,
            )
        )

    return candidates


# ---------------------------------------------------------------------
# Public search entry point
# ---------------------------------------------------------------------

def find_pdf_candidates(
    paper: Paper,
    max_results: int = 5,
    min_confidence: float = MIN_CONFIDENCE,
    include_rejected: bool = False,
) -> list[PdfCandidate] | tuple[list[PdfCandidate], list[RejectedCandidate]]:
    """
    Searches Unpaywall, Crossref, Semantic Scholar, arXiv, OpenAlex,
    and (if configured) Google Scholar via SerpApi for a legally
    open-access PDF matching this paper, and returns candidates ranked
    by (title + year + author) confidence (highest first), deduplicated
    by URL. Never downloads anything -- that only happens after the
    user confirms a candidate via download_and_attach_pdf().

    Set include_rejected=True to also get back every candidate that
    was found but scored below min_confidence, each tagged with why --
    useful for figuring out why a particular paper isn't returning
    anything.
    """

    if not paper.doi:
        resolved_doi = _resolve_doi_via_crossref(paper)
        if resolved_doi:
            paper.doi = resolved_doi

    all_candidates: list[PdfCandidate] = []
    all_candidates.extend(_search_unpaywall(paper))
    all_candidates.extend(_search_crossref_pdf_links(paper))
    all_candidates.extend(_search_semantic_scholar(paper))
    all_candidates.extend(_search_arxiv(paper))
    all_candidates.extend(_search_openalex(paper))
    all_candidates.extend(_search_google_scholar_via_serpapi(paper))

    if not all_candidates:
        logger.debug(
            "[find_pdf_candidates] every source returned zero raw candidates "
            "for paper id=%s title=%r -- check DEBUG logs above per source",
            paper.id, paper.title,
        )

    seen_urls: set[str] = set()
    accepted: list[PdfCandidate] = []
    rejected: list[RejectedCandidate] = []

    for candidate in all_candidates:
        if candidate.url in seen_urls:
            continue
        seen_urls.add(candidate.url)

        if candidate.confidence < min_confidence:
            rejected.append(
                RejectedCandidate(
                    url=candidate.url,
                    source=candidate.source,
                    title=candidate.title,
                    confidence=candidate.confidence,
                    reason=(
                        f"confidence {candidate.confidence:.2f} below "
                        f"threshold {min_confidence:.2f}"
                    ),
                )
            )
            continue

        accepted.append(candidate)

    accepted.sort(key=lambda c: c.confidence, reverse=True)
    rejected.sort(key=lambda c: c.confidence, reverse=True)

    if include_rejected:
        return accepted[:max_results], rejected

    return accepted[:max_results]


# ---------------------------------------------------------------------
# DOI-in-text verification (used by download_and_attach_pdf)
# ---------------------------------------------------------------------

def _normalize_doi(doi: str) -> str:
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi


def _pdf_contains_doi(pdf_path: Path, expected_doi: str) -> bool:
    """
    Scans the first few pages of a downloaded PDF's text for the
    paper's known DOI.
    """
    target = _normalize_doi(expected_doi)
    if not target:
        return False

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:DOI_SCAN_MAX_PAGES]:
                text = (page.extract_text() or "").lower()
                if target in text:
                    return True
    except Exception as exc:
        logger.debug("[verify] DOI-in-text scan failed: %s", exc)
        return False

    return False


# ---------------------------------------------------------------------
# Download + attach (only called after the user confirms a candidate)
# ---------------------------------------------------------------------

def download_and_attach_pdf(
    paper_id: int,
    url: str,
    expected_title: str | None = None,
    expected_doi: str | None = None,
) -> str:
    """
    Downloads the PDF at `url` and stores it the same way an uploaded
    PDF is stored (storage/papers/{paper_id}.pdf), returning the
    relative stored_path to save on Paper.stored_path.
    """

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            stream=True,
            headers={"User-Agent": "PaperRec/1.0 (academic PDF fetch)"},
        )
    except requests.RequestException as exc:
        raise ValueError(f"Could not reach that link: {exc}") from exc

    if not response.ok:
        raise ValueError(f"That link returned HTTP {response.status_code}.")

    content_type = response.headers.get("Content-Type", "").lower()

    tmp_path: Path | None = None

    try:
        with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            total_bytes = 0

            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > MAX_PDF_BYTES:
                    raise ValueError("The file at that link is too large.")
                tmp.write(chunk)

        is_pdf_header = "pdf" in content_type
        is_pdf_magic = tmp_path.read_bytes()[:5] == b"%PDF-"

        if not (is_pdf_header or is_pdf_magic):
            raise ValueError("That link did not return a PDF file.")

        doi_confirmed = False

        if expected_doi:
            doi_confirmed = _pdf_contains_doi(tmp_path, expected_doi)

        if not doi_confirmed and expected_title:
            try:
                extracted = extract_metadata_from_pdf(str(tmp_path))
                extracted_title = extracted.get("title")
            except Exception as exc:
                logger.debug("[verify] title re-extraction failed: %s", exc)
                extracted_title = None

            if extracted_title:
                match_score = _title_similarity(expected_title, extracted_title)

                if match_score < MIN_DOWNLOAD_TITLE_MATCH:
                    raise ValueError(
                        "This PDF doesn't appear to match the paper's title "
                        "-- it may be the wrong paper. Try another candidate "
                        "or upload the PDF manually."
                    )

        return save_paper_file(paper_id=paper_id, source_path=str(tmp_path))

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
