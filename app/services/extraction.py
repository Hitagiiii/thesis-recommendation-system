"""
PDF metadata auto-extraction with hybrid keyword extraction.

Extraction order:
1. Explicit author-provided keywords:
   - Keywords:
   - Index Terms:
   - Key words:
   - Subject terms:
2. YAKE-generated keywords from title + abstract.
3. None if no usable source text is available.

The keyword extraction uses two redundancy-control layers:

1. YAKE's built-in deduplication.
2. Additional normalized-word comparison to remove:
   - Singular/plural duplicates
   - Inflectional duplicates
   - Phrase containment
   - Highly overlapping keyword phrases

DOI extraction (new):
    Scans the extracted PDF text directly for a DOI, the same "read
    what's already on the document before searching external
    databases" approach Zotero's PDF metadata recognizer uses
    (recognizePDF.js looks for a DOI/ISBN in the extracted text and
    only falls back to a bibliographic search if none is found). A
    DOI found this way is far more reliable than the fuzzy
    title-similarity search app/services/pdf_finder.py otherwise has
    to fall back on (_resolve_doi_via_crossref), and it costs nothing
    -- no network call, just a regex over text already extracted for
    title/abstract/keywords.

Requires:
    pip install pdfplumber yake
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pdfplumber
import yake


# ---------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------

YEAR_PATTERN = re.compile(
    r"\b(19[5-9]\d|20[0-4]\d)\b"
)

ABSTRACT_HEADER = re.compile(
    r"^\s*abstract\b",
    re.IGNORECASE | re.MULTILINE,
)

NEXT_SECTION_HEADER = re.compile(
    r"\b("
    r"keywords|index terms|key words|subject terms|"
    r"1\.?\s+introduction|i\.\s+introduction|introduction|"
    r"table of contents|list of figures|list of tables|"
    r"acknowledge?ments?"
    r")\b",
    re.IGNORECASE,
)

KEYWORDS_LINE = re.compile(
    r"^\s*(?:"
    r"keywords?|"
    r"index\s+terms?|"
    r"key\s+words?|"
    r"subject\s+terms?"
    r")\s*"
    r"[:\-–—]\s*"
    r"(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

DOT_LEADER_PATTERN = re.compile(
    r"(?:\.[ \t]?){4,}\d+"
)

# Matches the DOI syntax registered with the International DOI
# Foundation: a "10." prefix, a 4+ digit registrant code, a slash,
# then a publisher-defined suffix. The suffix character class is
# intentionally broad (DOIs can legally contain many punctuation
# characters) but excludes whitespace and angle/quote characters that
# would mean the match has run into surrounding prose rather than the
# DOI itself.
DOI_PATTERN = re.compile(
    r"\b10\.\d{4,9}/[^\s\"'<>]+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Keyword cleaning configuration
# ---------------------------------------------------------------------

STOP_KEYWORD_PHRASES = {
    "keywords",
    "keyword",
    "index terms",
    "index term",
    "key words",
    "key word",
    "subject terms",
    "subject term",
    "abstract",
    "introduction",
    "references",
    "conclusion",
    "acknowledgment",
    "acknowledgement",
}


# Ordered suffix rules used only for redundancy comparison.
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("ization", "ize"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("ies", "y"),
    ("ing", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)


# Two phrases are considered redundant when their normalized
# word-root sets have at least this much Jaccard similarity.
_OVERLAP_REDUNDANCY_THRESHOLD = 0.75


# Common English words that usually indicate a sentence fragment rather than
# a meaningful academic keyword. These are used for YAKE output only.
_YAKE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "using", "we", "with", "our", "study", "studies",
    "paper", "propose", "proposed", "show", "shows", "based",
}


# ---------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------

def _extract_full_text(
    pdf_path: str,
    max_pages: int = 5,
) -> list[str]:
    """
    Return text from the first max_pages pages.

    Note:
        Scanned/image-only PDFs require OCR because pdfplumber
        cannot extract text from image-only pages.
    """
    pages_text: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            pages_text.append(
                page.extract_text() or ""
            )

    return pages_text


# ---------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------

def _extract_title(
    pdf_path: str,
    first_page_text: str,
) -> str | None:
    """
    Heuristic title extraction.

    Uses the largest text near the top of page 1.
    Falls back to the first substantial non-empty line.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return None

            page = pdf.pages[0]

            words = page.extract_words(
                extra_attrs=["size"]
            )

            if words:
                top_cutoff = page.height / 3

                candidates = [
                    word
                    for word in words
                    if word["top"] <= top_cutoff
                ]

                if candidates:
                    max_size = max(
                        word["size"]
                        for word in candidates
                    )

                    title_words = [
                        word["text"]
                        for word in candidates
                        if word["size"] >= max_size - 0.5
                    ]

                    title = " ".join(
                        title_words
                    ).strip()

                    if len(title) >= 8:
                        return title

    except Exception:
        # Fall back to text-based extraction if the
        # font-size heuristic fails.
        pass

    for line in first_page_text.splitlines():
        line = line.strip()

        if len(line) >= 8:
            return line

    return None


# ---------------------------------------------------------------------
# Abstract extraction
# ---------------------------------------------------------------------

def _is_toc_entry_line(
    text: str,
    position: int,
) -> bool:
    """
    Detect whether a matched heading is a Table of Contents entry.
    """
    line_start = text.rfind(
        "\n",
        0,
        position,
    ) + 1

    line_end = text.find(
        "\n",
        position,
    )

    if line_end == -1:
        line_end = len(text)

    line = text[line_start:line_end]

    return bool(
        DOT_LEADER_PATTERN.search(line)
    )


def _extract_abstract(
    full_text: str,
) -> str | None:
    """
    Extract text between the Abstract heading
    and the next section heading.
    """
    for match in ABSTRACT_HEADER.finditer(full_text):
        if _is_toc_entry_line(
            full_text,
            match.start(),
        ):
            continue

        after_header = full_text[
            match.end():
        ]

        next_match = NEXT_SECTION_HEADER.search(
            after_header
        )

        abstract_block = (
            after_header[:next_match.start()]
            if next_match
            else after_header[:2000]
        )

        # Skip likely Table of Contents content.
        if len(
            DOT_LEADER_PATTERN.findall(
                abstract_block
            )
        ) >= 2:
            continue

        abstract = " ".join(
            abstract_block.split()
        )

        abstract = abstract.strip(
            " :.-"
        )

        if len(abstract) >= 40:
            return abstract

    return None


# ---------------------------------------------------------------------
# DOI extraction (Zotero-style: read it off the document first)
# ---------------------------------------------------------------------

def _extract_doi(
    full_text: str,
) -> str | None:
    """
    Scans the extracted PDF text for a DOI.

    Academic PDFs very commonly print their own DOI on the title page
    or in a running header/footer, e.g.:

        doi: 10.1000/xyz123
        DOI:10.1000/xyz123
        https://doi.org/10.1000/xyz123

    Reading it directly here means:

        - upload_paper.py's Paper record gets a DOI at insertion time
          instead of staying doi=None until someone runs "Find PDF
          Online".
        - app/services/pdf_finder.py's find_pdf_candidates() then goes
          straight to Unpaywall (its strongest, DOI-keyed source) with
          high confidence, instead of first needing a fuzzy
          Crossref-by-title lookup (_resolve_doi_via_crossref) that
          can pick the wrong paper when titles are similar.

    Returns None if nothing plausible is found -- callers should treat
    that as "no DOI available yet", not as an error; pdf_finder.py's
    existing Crossref fallback still covers this case, and BibTeX
    imports (which have no PDF text at all) are unaffected.
    """
    match = DOI_PATTERN.search(full_text)

    if not match:
        return None

    doi = match.group(0)

    # Trim trailing characters that are almost always sentence/line
    # punctuation picked up by the greedy suffix match rather than
    # part of the DOI itself (a DOI can legally contain some of these
    # mid-string, but PDF-extracted text essentially never ends a DOI
    # on one of them).
    doi = doi.rstrip(").,;:]}\u2019\u201d")

    return doi or None


# ---------------------------------------------------------------------
# Keyword cleaning
# ---------------------------------------------------------------------

def _clean_keyword_phrase(
    keyword: str,
    *,
    reject_stopwords: bool = False,
) -> str | None:
    """
    Clean one keyword phrase.

    Explicit author keywords keep the original permissive behavior. YAKE
    keywords additionally reject sentence fragments and generic stopwords.
    """
    keyword = keyword.strip().lower()

    # Normalize hyphens/dashes so equivalent phrases compare consistently.
    keyword = re.sub(r"[-‐-‒–—―]", " ", keyword)

    # Remove punctuation from the beginning/end and normalize whitespace.
    keyword = keyword.strip(" \t\r\n:;,.•·()[]{}")
    keyword = re.sub(r"\s+", " ", keyword)

    if keyword in STOP_KEYWORD_PHRASES:
        return None

    if len(keyword) < 2 or len(keyword) > 100:
        return None

    alphabetic_count = sum(character.isalpha() for character in keyword)
    if alphabetic_count < 2:
        return None

    if reject_stopwords:
        words = keyword.split()

        # Reject sentence-like YAKE fragments such as ``completion we study``.
        if any(word in _YAKE_STOPWORDS for word in words):
            return None

        # YAKE phrases should contain at least two meaningful words.
        meaningful_words = [
            word for word in words
            if word not in _YAKE_STOPWORDS and len(word) > 1
        ]
        if len(meaningful_words) < 2 or len(meaningful_words) > 6:
            return None

    return keyword


def _normalize_word(
    word: str,
) -> str:
    """
    Apply light, dependency-free normalization.

    This is used only to compare keyword phrases for redundancy.
    It is not used for the final displayed/stored keyword text.
    """
    word = word.lower()

    # Avoid over-stemming short words.
    if len(word) <= 4:
        return word

    for suffix, replacement in _SUFFIX_RULES:
        if (
            word.endswith(suffix)
            and len(word) - len(suffix) >= 3
        ):
            return (
                word[: -len(suffix)]
                + replacement
            )

    return word


def _normalized_word_set(
    phrase: str,
) -> frozenset[str]:
    """
    Return normalized word roots for a keyword phrase.
    """
    return frozenset(
        _normalize_word(word)
        for word in phrase.split()
    )


def _is_redundant_keyword(
    candidate: str,
    selected_keywords: list[str],
) -> bool:
    """
    Detect redundant or overlapping keyword phrases.

    A candidate is considered redundant when:

    1. Its normalized word set is identical to a selected keyword.
    2. Its normalized word set is contained in a selected keyword.
    3. A selected keyword's normalized word set is contained
       in the candidate.
    4. Jaccard similarity is at least 0.6.
    """
    candidate_words = _normalized_word_set(
        candidate
    )

    if not candidate_words:
        return False

    for selected in selected_keywords:
        selected_words = _normalized_word_set(
            selected
        )

        if not selected_words:
            continue

        # Exact normalized match.
        if candidate_words == selected_words:
            return True

        # Phrase containment.
        if (
            candidate_words <= selected_words
            or selected_words <= candidate_words
        ):
            return True

        # Jaccard similarity.
        union = candidate_words | selected_words
        intersection = candidate_words & selected_words

        if union:
            similarity = (
                len(intersection)
                / len(union)
            )

            if (
                similarity
                >= _OVERLAP_REDUNDANCY_THRESHOLD
            ):
                return True

    return False


# ---------------------------------------------------------------------
# Explicit author-provided keywords
# ---------------------------------------------------------------------

def _split_explicit_keywords(
    raw: str,
) -> list[str]:
    """
    Split author-provided keywords.

    Supports:
    - Commas
    - Semicolons
    - Vertical bars
    - Bullets
    - New lines
    """
    parts = re.split(
        r"[,;|•·\n]+",
        raw,
    )

    cleaned: list[str] = []

    for part in parts:
        keyword = _clean_keyword_phrase(
            part
        )

        if not keyword:
            continue

        if _is_redundant_keyword(
            keyword,
            cleaned,
        ):
            continue

        cleaned.append(keyword)

    return cleaned


def _extract_explicit_keywords(
    full_text: str,
) -> list[str]:
    """
    Extract explicit author-provided keywords.

    Returns an empty list if no recognized keyword label
    is found.
    """
    match = KEYWORDS_LINE.search(
        full_text
    )

    if not match:
        return []

    raw = match.group(1)

    # Avoid capturing a large following paragraph.
    raw = raw.split(
        "\n\n"
    )[0]

    raw = raw.split(
        "\n"
    )[0]

    return _split_explicit_keywords(
        raw
    )


# ---------------------------------------------------------------------
# YAKE keyword extraction
# ---------------------------------------------------------------------

def _build_keyword_source_text(
    title: str | None,
    abstract: str | None,
) -> str:
    """
    Build the text used by YAKE.

    The title is repeated once to give it slightly more importance.
    """
    parts: list[str] = []

    if title:
        parts.append(title)
        parts.append(title)

    if abstract:
        parts.append(abstract)

    return " ".join(
        parts
    ).strip()


def _extract_yake_keywords(
    text: str,
    max_keywords: int = 8,
    max_ngram_size: int = 3,
) -> list[str]:
    """
    Generate cleaner keywords using YAKE.

    YAKE's built-in deduplication handles near-identical
    surface forms. The additional redundancy check handles:

    - Singular/plural variants
    - Inflectional variants
    - Phrase containment
    - Highly overlapping phrases
    """
    if not text.strip():
        return []

    extractor = yake.KeywordExtractor(
        lan="en",
        n=max_ngram_size,
        top=max(30, max_keywords * 8),
        dedupLim=0.85,
        dedupFunc="seqm",
        windowsSize=2,
    )

    ranked_keywords = extractor.extract_keywords(
        text
    )

    selected_keywords: list[str] = []
    seen: set[str] = set()

    for keyword, _score in ranked_keywords:
        cleaned_keyword = _clean_keyword_phrase(
            keyword,
            reject_stopwords=True,
        )

        if not cleaned_keyword:
            continue

        if cleaned_keyword in seen:
            continue

        if _is_redundant_keyword(
            cleaned_keyword,
            selected_keywords,
        ):
            continue

        selected_keywords.append(
            cleaned_keyword
        )

        seen.add(
            cleaned_keyword
        )

        if len(selected_keywords) >= max_keywords:
            break

    return selected_keywords


# ---------------------------------------------------------------------
# Hybrid keyword extraction
# ---------------------------------------------------------------------

def _keywords_to_string(
    keywords: list[str],
) -> str | None:
    """
    Convert a keyword list to database string format.
    """
    if not keywords:
        return None

    return ", ".join(
        keywords
    )


def _extract_keywords_hybrid(
    full_text: str,
    title: str | None,
    abstract: str | None,
) -> dict[str, Any]:
    """
    Hybrid keyword extraction.

    Priority:

    1. Explicit author-provided keywords.
    2. YAKE-generated keywords.
    3. None if no usable keywords are available.

    Returns:
        {
            "keywords": str | None,
            "keywords_source": "author" | "yake" | None,
            "keywords_generated": bool,
        }
    """

    # ---------------------------------------------------------
    # 1. Prefer explicit author-provided keywords
    # ---------------------------------------------------------
    explicit_keywords = _extract_explicit_keywords(
        full_text
    )

    if explicit_keywords:
        return {
            "keywords": _keywords_to_string(
                explicit_keywords
            ),
            "keywords_source": "author",
            "keywords_generated": False,
        }

    # ---------------------------------------------------------
    # 2. Fall back to YAKE using title + abstract
    # ---------------------------------------------------------
    source_text = _build_keyword_source_text(
        title=title,
        abstract=abstract,
    )

    yake_keywords = _extract_yake_keywords(
        text=source_text,
        max_keywords=8,
        max_ngram_size=3,
    )

    if yake_keywords:
        return {
            "keywords": _keywords_to_string(
                yake_keywords
            ),
            "keywords_source": "yake",
            "keywords_generated": True,
        }

    # ---------------------------------------------------------
    # 3. No usable keywords could be generated
    # ---------------------------------------------------------
    return {
        "keywords": None,
        "keywords_source": None,
        "keywords_generated": False,
    }


# ---------------------------------------------------------------------
# Publication year extraction
# ---------------------------------------------------------------------

def _extract_publication_year(
    full_text: str,
) -> int | None:
    """
    Extract the most frequent plausible publication year.

    Only years from 1950 through 2049 are considered.
    Future years beyond the current year are ignored.
    """
    current_year = datetime.now().year

    years = [
        int(year)
        for year in YEAR_PATTERN.findall(
            full_text
        )
        if int(year) <= current_year
    ]

    if not years:
        return None

    return max(
        set(years),
        key=years.count,
    )


# ---------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------

def extract_metadata_from_pdf(
    pdf_path: str,
) -> dict[str, Any]:
    """
    Extract metadata from a PDF.

    Returns:
        {
            "title": str | None,
            "abstract": str | None,
            "keywords": str | None,
            "keywords_source": "author" | "yake" | None,
            "keywords_generated": bool,
            "publication_year": int | None,
            "doi": str | None,
        }
    """
    pages_text = _extract_full_text(
        pdf_path=pdf_path,
        max_pages=5,
    )

    first_page_text = (
        pages_text[0]
        if pages_text
        else ""
    )

    full_text = "\n".join(
        pages_text
    )

    title = _extract_title(
        pdf_path=pdf_path,
        first_page_text=first_page_text,
    )

    abstract = _extract_abstract(
        full_text=full_text,
    )

    keyword_result = _extract_keywords_hybrid(
        full_text=full_text,
        title=title,
        abstract=abstract,
    )

    return {
        "title": title,
        "abstract": abstract,
        "keywords": keyword_result[
            "keywords"
        ],
        "keywords_source": keyword_result[
            "keywords_source"
        ],
        "keywords_generated": keyword_result[
            "keywords_generated"
        ],
        "publication_year": _extract_publication_year(
            full_text
        ),
        "doi": _extract_doi(
            full_text
        ),
    }
