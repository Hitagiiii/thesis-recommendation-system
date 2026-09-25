"""
Fills in a paper's missing/corrupted Title, Abstract, Publication Year,
and Author using candidates already retrieved by PDF discovery
(app/services/pdf_finder.py) -- data those search functions already
download to compute a confidence score, then previously discarded once
scoring was done.

Design constraints (why each rule exists):

    - Only ever REPLACE a field that is missing or looks broken.
      Never touch a field that already looks fine. A false-positive
      title/author match on an otherwise-correct paper is a worse
      outcome than leaving a field blank.
    - Each field uses its own confidence bar. Title replacement reuses
      the 0.72 threshold this codebase already trusts elsewhere
      (pdf_finder.MIN_CROSSREF_DOI_MATCH) for "confident enough to
      overwrite something"; abstract/year/author use the looser
      pdf_finder.MIN_CONFIDENCE bar already used to accept a candidate
      into the results list at all, since filling a *blank* field is
      lower-risk than overwriting a populated one.
    - Keywords are deliberately NOT enriched here -- see the note in
      enrich_paper_metadata()'s docstring.
"""

from __future__ import annotations

import logging

from app.models.models import Paper
from app.services.pdf_finder import MIN_CONFIDENCE, PdfCandidate

logger = logging.getLogger(__name__)

# Same bar this codebase already uses elsewhere (pdf_finder.py's
# MIN_CROSSREF_DOI_MATCH) for "confident enough to let external data
# overwrite something already on the record" -- title replacement is
# the one enrichment case here that overwrites a populated field
# rather than filling a blank one, so it gets the stricter bar.
MIN_TITLE_REPLACEMENT_CONFIDENCE = 0.72

# Mirrors validate_paper()'s abstract-length requirement -- an
# abstract shorter than this wouldn't make the paper valid anyway.
MIN_ABSTRACT_LENGTH = 40

# Corrupted extracted titles look like "]oc.htam[" -- a garbled,
# reversed-and-bracketed string some PDF extractors produce on certain
# layouts. Mirrors the detection already used to exclude such titles
# from classification text in app/services/classification.py, kept in
# sync here rather than imported so this module has no dependency on
# classification.py's internals.
_KNOWN_CORRUPTED_TITLES = {
    "]oc.htam[",
    "]an.htam[",
    "]ts.htam[",
    "]hp-oib.scisyhp[",
    "]ac.htam[",
    "]co.htam[",
    "]af.htam[",
}


def is_title_corrupted(title: str | None) -> bool:
    """
    True when a title is missing or matches the garbled-bracket
    pattern this codebase already treats as unusable elsewhere.
    """
    if not title:
        return True

    clean = title.strip()

    if not clean:
        return True

    return (
        clean in _KNOWN_CORRUPTED_TITLES
        or (clean.startswith("]") and clean.endswith("["))
        or (clean.count("]") >= 1 and clean.count("[") >= 1)
    )


def _best_candidate_for(
    candidates: list[PdfCandidate],
    *,
    requires: str,
) -> PdfCandidate | None:
    """
    Highest-confidence candidate (at or above MIN_CONFIDENCE) that has
    a usable value for the given field ("abstract" | "publication_year"
    | "authors"), or None if no candidate qualifies.
    """
    usable = [
        c
        for c in candidates
        if c.confidence >= MIN_CONFIDENCE and getattr(c, requires, None)
    ]

    if requires == "abstract":
        usable = [
            c for c in usable
            if len((c.abstract or "").strip()) >= MIN_ABSTRACT_LENGTH
        ]

    if not usable:
        return None

    return max(usable, key=lambda c: c.confidence)


def enrich_paper_metadata(
    paper: Paper,
    candidates: list[PdfCandidate],
) -> list[str]:
    """
    Fills Title (if corrupted/missing), Abstract (if missing or too
    short to be valid), Publication Year (if missing), and Author (if
    missing) using the PDF-discovery candidates already found for this
    paper -- each field independently, from whichever candidate is the
    best match *for that field*, not necessarily the single overall
    best candidate.

    Keywords are intentionally left alone: none of the sources
    pdf_finder.py queries return keywords in a form comparable to this
    project's author/YAKE keyword field (OpenAlex's "concepts" are a
    different taxonomy, not a drop-in replacement), so merging them
    risks polluting keyword-based classification/TF-IDF text for
    little benefit.

    Returns the list of field names actually changed (e.g.
    ["title", "abstract"]), so callers know whether to re-run
    validate_paper() / refresh_prepared_text(). Also sets
    paper.enrichment_notes to a short, human-readable trace of what
    changed and from which source -- purely for auditability (e.g. for
    a methodology write-up), never read by any other part of the
    system.
    """
    if not candidates:
        return []

    changed: list[str] = []
    notes: list[str] = []

    # ---------------------------------------------------------
    # Title -- the only field here that OVERWRITES something
    # already present, so it uses the stricter confidence bar.
    # ---------------------------------------------------------
    if is_title_corrupted(paper.title):
        title_candidates = [c for c in candidates if c.title]
        if title_candidates:
            best_title_candidate = max(title_candidates, key=lambda c: c.confidence)

            if best_title_candidate.confidence >= MIN_TITLE_REPLACEMENT_CONFIDENCE:
                paper.title = best_title_candidate.title
                changed.append("title")
                notes.append(
                    f"title <- {best_title_candidate.source} "
                    f"(confidence {best_title_candidate.confidence:.2f})"
                )

    # ---------------------------------------------------------
    # Abstract
    # ---------------------------------------------------------
    needs_abstract = (
        not paper.abstract
        or len(paper.abstract.strip()) < MIN_ABSTRACT_LENGTH
    )

    if needs_abstract:
        candidate = _best_candidate_for(candidates, requires="abstract")
        if candidate:
            paper.abstract = candidate.abstract.strip()
            changed.append("abstract")
            notes.append(
                f"abstract <- {candidate.source} "
                f"(confidence {candidate.confidence:.2f})"
            )

    # ---------------------------------------------------------
    # Publication year
    # ---------------------------------------------------------
    if paper.publication_year is None:
        candidate = _best_candidate_for(candidates, requires="publication_year")
        if candidate:
            paper.publication_year = candidate.publication_year
            changed.append("publication_year")
            notes.append(
                f"publication_year <- {candidate.source} "
                f"(confidence {candidate.confidence:.2f})"
            )

    # ---------------------------------------------------------
    # Author
    # ---------------------------------------------------------
    if not paper.author:
        candidate = _best_candidate_for(candidates, requires="authors")
        if candidate:
            paper.author = " and ".join(candidate.authors)
            changed.append("author")
            notes.append(
                f"author <- {candidate.source} "
                f"(confidence {candidate.confidence:.2f})"
            )

    if changed:
        logger.info(
            "[enrichment] paper id=%s enriched fields=%s",
            getattr(paper, "id", None), changed,
        )

        # Optional column -- see scripts/migrate_add_enrichment_notes.py.
        # Guarded with hasattr so this module still works against a
        # Paper model that hasn't had the migration applied yet.
        if hasattr(paper, "enrichment_notes"):
            existing = (paper.enrichment_notes or "").strip()
            new_note = "; ".join(notes)
            paper.enrichment_notes = (
                f"{existing}; {new_note}" if existing else new_note
            )

    return changed
