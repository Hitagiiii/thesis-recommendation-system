"""
Runs metadata enrichment (app/services/metadata_enrichment.py) against
every paper ALREADY in the database, not just new uploads.

This is what turns the "more thorough" enrichment path into a
countable before/after number for a methodology chapter: run this
once, and the console summary tells you how many previously-invalid
papers became valid-for-recommendation.

Only touches papers that _needs_enrichment() flags as incomplete
(missing/short abstract, missing year, missing author, or a corrupted
title) -- an already-complete paper is skipped and costs nothing.

Safe to re-run. Does NOT rebuild TF-IDF/S-BERT vectors itself -- run
scripts/rebuild_recommendation.py afterward if any paper's validity
changed, so the recommendation index picks up the newly-valid papers.

Usage:
    python -m scripts.enrich_existing_papers
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import SessionLocal
from app.models.models import Paper
from app.services.pdf_finder import find_pdf_candidates
from app.services.metadata_enrichment import enrich_paper_metadata

from app.services.validation import validate_paper
from app.services.text_preparation import refresh_prepared_text

def _needs_enrichment(paper: Paper) -> bool:
    """
    Return True when a paper has metadata that may be improved
    through external metadata/PDF enrichment.

    Papers are considered candidates when they have:
    - no title or a suspiciously short title
    - no author
    - no abstract or a very short abstract
    - no publication year
    """

    title = (paper.title or "").strip()
    author = (paper.author or "").strip()
    abstract = (paper.abstract or "").strip()

    # Missing or suspiciously short title.
    if not title or len(title) < 3:
        return True

    # Missing author.
    if not author:
        return True

    # Missing or very short abstract.
    if not abstract or len(abstract) < 40:
        return True

    # Missing publication year.
    if paper.publication_year is None:
        return True

    return False
def main() -> None:
    db = SessionLocal()

    try:
        papers = db.query(Paper).all()

        candidates_for_enrichment = [p for p in papers if _needs_enrichment(p)]

        print(f"Total papers in repository: {len(papers)}")
        print(f"Papers flagged for enrichment: {len(candidates_for_enrichment)}")
        print()

        invalid_before = sum(
            1 for p in papers if not p.is_valid_for_recommendation
        )

        enriched_count = 0
        newly_valid_count = 0

        for index, paper in enumerate(candidates_for_enrichment, start=1):
            print(f"[{index}/{len(candidates_for_enrichment)}] "
                  f"id={paper.id} {paper.title!r}")

            if not paper.title:
                print("  [SKIP] No title to search with.")
                continue

            was_valid = paper.is_valid_for_recommendation

            try:
                candidates = find_pdf_candidates(paper)
                changed = enrich_paper_metadata(paper, candidates)
            except Exception as exc:
                print(f"  [ERROR] {exc}")
                continue

            if not changed:
                print("  [NO CHANGE] Nothing enrichable found.")
                continue

            validate_paper(paper)
            refresh_prepared_text(paper)

            db.commit()
            db.refresh(paper)

            enriched_count += 1
            print(f"  [ENRICHED] fields={changed}")

            if not was_valid and paper.is_valid_for_recommendation:
                newly_valid_count += 1
                print("  [NOW VALID] This paper is now valid-for-recommendation.")

        invalid_after = sum(
            1 for p in db.query(Paper).all() if not p.is_valid_for_recommendation
        )

        print()
        print("=" * 60)
        print("ENRICHMENT COMPLETE")
        print("=" * 60)
        print(f"Papers enriched:            {enriched_count}")
        print(f"Papers newly valid:         {newly_valid_count}")
        print(f"Invalid papers before:      {invalid_before}")
        print(f"Invalid papers after:       {invalid_after}")
        print()
        print("Run 'python -m scripts.rebuild_recommendation' next if any")
        print("paper's validity changed, so TF-IDF/S-BERT pick it up.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
