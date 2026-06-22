"""Pure text-similarity helpers (no Apify/network deps, so they are unit-testable).

Used by both ``dedup`` (collapse same-story articles) and ``ai_groq`` (collapse
same-subject selections). Matching is accent/punctuation-insensitive and ignores a
small pt-BR stopword set so that two headlines about the same subject — even worded
very differently — share their significant tokens.
"""

from __future__ import annotations

import re
import unicodedata

# Tiny pt-BR stopword set so common words don't inflate similarity.
STOPWORDS = {
    'a', 'o', 'as', 'os', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da', 'dos', 'das',
    'e', 'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'pra', 'com', 'sem', 'que',
    'se', 'ao', 'aos', 'sobre', 'após', 'apos', 'entre', 'sua', 'seu', 'suas', 'seus',
    'nova', 'novo', 'novas', 'novos', 'the', 'of', 'to', 'in', 'on', 'for', 'and',
}


def strip_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)
    )


def tokens(text: str) -> set[str]:
    """Significant, accent-free, lowercased tokens of a string."""
    clean = strip_accents((text or '').lower())
    words = re.findall(r'[a-z0-9]+', clean)
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def overlap_coefficient(a: set[str], b: set[str]) -> float:
    """Overlap coefficient: |A∩B| / min(|A|,|B|).

    More forgiving than Jaccard when one text is short (e.g. a 3-word subject label):
    a short subject fully contained in a longer one scores 1.0 instead of being diluted.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def same_subject(a: str, b: str, *, threshold: float = 0.6) -> bool:
    """True if two short subject labels refer to the same subject/event.

    Uses the overlap coefficient (good for short labels). ``threshold`` is the share
    of the smaller label's significant tokens that must appear in the other.
    """
    ta, tb = tokens(a), tokens(b)
    return overlap_coefficient(ta, tb) >= threshold
