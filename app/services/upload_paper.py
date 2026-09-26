from pathlib import Path

from sqlalchemy.orm import Session

from app.models.models import Paper

from app.services.bib_extraction import extract_metadata_from_bib
from app.services.classification import classify_paper
from app.services.extraction import extract_metadata_from_pdf
from app.services.text_preparation import refresh_prepared_text
from app.services.validation import validate_paper
from app.services.storage import save_paper_file
from app.services.pdf_finder import (
    find_pdf_candidates,
    download_and_attach_pdf,
    MIN_CONFIDENCE,
    PdfCandidate,
)
from app.services.metadata_enrichment import (
    enrich_paper_metadata,
    is_title_corrupted,
)


def _refresh_recommendation_fields(paper: Paper) -> None:
    """Prepare the text used by the recommendation pipeline."""
    try:
        refresh_prepared_text(paper)
    except Exception as exc:
        print(
            f"WARNING: Could not prepare recommendation text: {exc}"
        )


def _needs_enrichment(paper: Paper) -> bool:
    """
    True when at least one of the four recommendation-required fields
    is missing/unusable, or the title looks corrupted -- i.e. there's
    actually something for enrichment to fix. Used to skip the network
    round-trip entirely for papers that already extracted cleanly.
    """
    if is_title_corrupted(paper.title):
        return True

    if not paper.abstract or len(paper.abstract.strip()) < 40:
        return True

    if paper.publication_year is None:
        return True

    if not paper.author:
        return True

    return False


def _try_enrich_from_pdf_discovery(paper: Paper) -> list[PdfCandidate]:
    """
    Best-effort automatic metadata enrichment: runs the same PDF-
    discovery lookup used by "Find PDF Online" (Unpaywall, Crossref,
    Semantic Scholar, arXiv, OpenAlex) on every upload, right after
    validation, instead of only when a user manually clicks Find PDF.

    Returns the candidate list it found (possibly empty), so callers
    -- specifically _try_auto_attach_pdf() below -- can reuse it
    instead of searching again from scratch.

    Deliberately conservative:
        - Skipped entirely if the paper already looks complete
          (_needs_enrichment), so a clean upload costs nothing extra.
        - Skipped if there's no title to search with.
        - Never raises. A network hiccup here must never fail the
          upload itself.

    validate_paper() and refresh_prepared_text() are re-run afterward
    only if enrich_paper_metadata() actually changed something.
    """
    if not _needs_enrichment(paper):
        return []

    if not paper.title:
        return []

    try:
        candidates = find_pdf_candidates(paper)
        changed_fields = enrich_paper_metadata(paper, candidates)

        if changed_fields:
            print(
                f"INFO: Metadata enrichment filled {changed_fields} "
                f"for {paper.source_filename!r}"
            )

            validate_paper(paper)
            refresh_prepared_text(paper)

        return candidates

    except Exception as exc:
        print(f"WARNING: Metadata enrichment failed: {exc}")
        return []


def _try_auto_attach_pdf(
    paper: Paper,
    candidates: list[PdfCandidate],
) -> bool:
    """
    Best-effort automatic PDF attachment for papers that were imported
    from a citation only (BibTeX / Google Scholar URL) and therefore
    have no real PDF file -- only the original .bib text is stored.

    Reuses the candidate list PDF discovery already found during
    enrichment (_try_enrich_from_pdf_discovery), rather than searching
    again, since every source in pdf_finder.py was already queried
    once for this paper.

    Only attaches when:
        - the paper doesn't already have a stored PDF
        - at least one candidate exists
        - the best candidate meets the same confidence bar the manual
          "Find PDF Online" flow already trusts (MIN_CONFIDENCE)

    Requires paper.id to already exist (i.e. must run after the first
    commit), since download_and_attach_pdf() names the file
    "{paper_id}.pdf". Never raises -- same conservative contract as
    enrichment: a failed download must not break the upload, since the
    paper is already safely stored with its original .bib file.

    Returns True if a PDF was attached (caller should re-commit).
    """
    if paper.stored_path and paper.stored_path.lower().endswith(".pdf"):
        return False  # already has a real PDF

    if not candidates:
        return False

    # candidates is already sorted by confidence, descending
    # (see find_pdf_candidates() in pdf_finder.py)
    best = candidates[0]

    if best.confidence < MIN_CONFIDENCE:
        return False

    try:
        stored_path = download_and_attach_pdf(
            paper_id=paper.id,
            url=best.url,
            expected_title=paper.title,
            expected_doi=paper.doi,
        )

        paper.stored_path = stored_path

        print(
            f"INFO: Auto-attached PDF for paper id={paper.id} "
            f"from {best.source} (confidence {best.confidence:.2f})"
        )

        return True

    except Exception as exc:
        print(
            f"WARNING: Automatic PDF attachment failed for "
            f"paper id={paper.id}: {exc}"
        )
        return False


def upload_paper(
    db: Session,
    source_path: str,
    original_filename: str | None = None,
) -> Paper:
    """
    Import a PDF or BibTeX file into the repository.

    PDF:
        Uses the existing PDF extraction pipeline. extraction.py also
        reads a DOI directly off the PDF's own text.

    BibTeX:
        Uses the BibTeX parser, followed by best-effort metadata
        enrichment AND automatic PDF attachment when a confident
        open-access match exists -- so a citation-only import ends up
        with a real PDF file instead of just the original .bib text,
        whenever one can be found.

    Both:
        Followed by automatic metadata enrichment (Step 5b) that fills
        in whatever Title/Abstract/Publication Year/Author extraction
        couldn't recover, using PDF-discovery search results.
    """

    source = Path(source_path)

    if not source.exists():
        raise FileNotFoundError(
            f"Source file does not exist: {source}"
        )

    if not source.is_file():
        raise ValueError(
            f"Source path is not a file: {source}"
        )

    extension = source.suffix.lower()

    if extension not in {".pdf", ".bib"}:
        raise ValueError(
            "Only PDF and BibTeX (.bib) files are supported."
        )

    filename = original_filename or source.name

    # ---------------------------------------------------------
    # STEP 1 — Extract metadata
    # ---------------------------------------------------------

    if extension == ".pdf":
        metadata = extract_metadata_from_pdf(
            str(source)
        )
    else:
        metadata = extract_metadata_from_bib(
            str(source)
        )

    metadata = metadata or {}

    # ---------------------------------------------------------
    # STEP 2 — Create Paper
    # ---------------------------------------------------------

    paper = Paper(
        title=metadata.get("title"),
        author=metadata.get("author"),
        abstract=metadata.get("abstract"),
        keywords=metadata.get("keywords"),
        keywords_source=metadata.get(
            "keywords_source"
        ),
        keywords_generated=metadata.get(
            "keywords_generated",
            False,
        ),
        publication_year=metadata.get(
            "publication_year"
        ),
        doi=metadata.get("doi"),
        citation_count=metadata.get(
            "citation_count"
        ),
        source_filename=filename,
        extraction_method=(
            "bibtex"
            if extension == ".bib"
            else "pdf"
        ),
    )

    # ---------------------------------------------------------
    # STEP 4 — Classification
    # ---------------------------------------------------------

    try:
        classify_paper(paper)
    except Exception as exc:
        print(
            f"WARNING: Paper classification failed: {exc}"
        )

    # ---------------------------------------------------------
    # STEP 5 — Validation
    # ---------------------------------------------------------

    try:
        validate_paper(paper)
    except Exception as exc:
        print(
            f"WARNING: Paper validation failed: {exc}"
        )

    # ---------------------------------------------------------
    # STEP 5b — Automatic metadata enrichment ("thorough" path)
    #
    # Runs before the first commit so an enriched paper is written
    # to the database once, already enriched, rather than inserted
    # incomplete and patched in a second write.
    #
    # The candidates found here are kept (not discarded) so Step 8b
    # below can reuse them for automatic PDF attachment without a
    # second round of network calls.
    # ---------------------------------------------------------

    pdf_candidates = _try_enrich_from_pdf_discovery(paper)

    # ---------------------------------------------------------
    # STEP 6 — Prepare recommendation text
    # ---------------------------------------------------------

    _refresh_recommendation_fields(paper)

    # ---------------------------------------------------------
    # STEP 7 — Save database record
    #
    # paper.id is assigned here. Automatic PDF attachment (Step 8b)
    # must run after this, because download_and_attach_pdf() needs
    # paper.id to name the stored file.
    # ---------------------------------------------------------

    try:
        db.add(paper)
        db.commit()
        db.refresh(paper)

    except Exception:
        db.rollback()
        raise

    # ---------------------------------------------------------
    # STEP 8 — Save physical file (original source: PDF or .bib)
    # ---------------------------------------------------------

    try:
        stored_path = save_paper_file(
            paper_id=paper.id,
            source_path=str(source),
        )

        paper.stored_path = stored_path

        db.commit()
        db.refresh(paper)

    except Exception:
        db.rollback()

        try:
            db.delete(paper)
            db.commit()
        except Exception:
            db.rollback()

        raise

    # ---------------------------------------------------------
    # STEP 8b — Automatic PDF attachment
    #
    # Only relevant for citation-only imports (.bib / Google Scholar):
    # if PDF discovery already found a confident open-access match
    # while enriching metadata (Step 5b), download it now and replace
    # the stored .bib pointer with a real PDF file. Best-effort and
    # non-fatal -- the paper already has a valid stored_path (the
    # .bib file) even if this fails.
    # ---------------------------------------------------------

    if extension == ".bib":
        try:
            attached = _try_auto_attach_pdf(paper, pdf_candidates)

            if attached:
                db.commit()
                db.refresh(paper)

        except Exception as exc:
            db.rollback()
            print(
                f"WARNING: Auto-attach step failed for paper "
                f"id={paper.id}: {exc}"
            )

    return paper


def upload_paper_from_pdf(
    db: Session,
    source_path: str,
    original_filename: str | None = None,
) -> Paper:
    """
    Backward-compatible wrapper for the existing API.

    The API historically called this function for PDF uploads.
    The underlying upload_paper() function now handles both
    PDF and BibTeX files.
    """

    return upload_paper(
        db=db,
        source_path=source_path,
        original_filename=original_filename,
    )


def complete_paper_manually(
    db: Session,
    paper: Paper,
    **updates,
) -> Paper:
    """
    Backward-compatible helper for manually completing
    paper metadata.

    Only supported Paper metadata fields are updated.
    Classification, validation, and recommendation text
    are refreshed after the update.
    """

    allowed_fields = {
        "title",
        "author",
        "abstract",
        "keywords",
        "publication_year",
        "doi",
        "subject_category",
        "document_type",
        "citation_count",
    }

    for field, value in updates.items():
        if field in allowed_fields:
            setattr(paper, field, value)

    # ---------------------------------------------------------
    # Re-run classification after metadata changes
    # ---------------------------------------------------------

    try:
        classify_paper(paper)
    except Exception as exc:
        print(
            f"WARNING: Paper classification failed: {exc}"
        )

    # ---------------------------------------------------------
    # Re-run validation after metadata changes
    # ---------------------------------------------------------

    try:
        validate_paper(paper)
    except Exception as exc:
        print(
            f"WARNING: Paper validation failed: {exc}"
        )

    # ---------------------------------------------------------
    # Rebuild recommendation text
    # ---------------------------------------------------------

    _refresh_recommendation_fields(paper)

    # ---------------------------------------------------------
    # Save changes
    # ---------------------------------------------------------

    try:
        db.add(paper)
        db.commit()
        db.refresh(paper)

    except Exception:
        db.rollback()
        raise

    return paper