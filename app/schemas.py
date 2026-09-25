"""
Pydantic schemas -- the shapes of data going in and out of the API.
Kept separate from models.py (the database shapes) so the two can
diverge if needed (e.g. hiding password_hash from API responses).
"""

from datetime import datetime
from pydantic import BaseModel


class PaperOut(BaseModel):
    id: int
    title: str
    author: str | None = None
    abstract: str | None = None
    keywords: str | None = None
    publication_year: int | None = None
    doi: str | None = None
    subject_category: str | None = None
    document_type: str | None = None
    citation_count: int | None = None
    is_valid_for_recommendation: bool
    missing_fields: str | None = None
    enrichment_notes: str | None = None   
    source_filename: str | None = None
    extraction_method: str | None = None
    stored_path: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True  # lets this build directly from a SQLAlchemy Paper


class PaperUpdate(BaseModel):
    """Fields a user can fill in manually -- e.g. completing missing ones
    after an incomplete auto-extraction, or adding the ones extraction
    never touches (author, doi, subject_category, document_type,
    citation_count)."""

    title: str | None = None
    author: str | None = None
    abstract: str | None = None
    keywords: str | None = None
    publication_year: int | None = None
    doi: str | None = None
    subject_category: str | None = None
    document_type: str | None = None
    citation_count: int | None = None


class RepositoryStats(BaseModel):
    total_papers: int
    by_subject: dict[str, int]
    category_count: int


class LibraryEntryOut(BaseModel):
    paper: PaperOut
    saved_at: datetime


class SearchResultOut(BaseModel):
    """One ranked result from /api/recommendations."""

    paper: PaperOut
    score: float


class PdfCandidateOut(BaseModel):
    """One candidate PDF returned by GET /api/papers/{id}/find-pdf.

    Nothing has been downloaded yet at this point -- these are just
    search results (title, source, confidence, links) for the user to
    review and confirm before anything is attached.
    """

    url: str
    source: str  # "unpaywall" | "semantic_scholar" | "arxiv"
    title: str | None = None
    confidence: float
    landing_page_url: str | None = None
    license: str | None = None


class AttachPdfRequest(BaseModel):
    """Body for POST /api/papers/{id}/attach-pdf -- the URL the user
    confirmed from a PdfCandidateOut."""

    url: str
