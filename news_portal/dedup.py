"""Deduplication so the portal never republishes the same news.

Two layers, both best-effort (any failure is logged and the run continues):

1. In-run: collapse articles that cover the SAME story but came from different
   sources/URLs, keeping the first (highest-ranked) one. This complements the
   editor AI's own dedup with a deterministic safety net.
2. Across runs: a persistent history (a *named* key-value store that survives
   between Actor runs) of everything already published, so a story picked up in a
   previous run is never selected again.

Matching is done on a normalized fingerprint of the URL plus a token signature of
the title (accent/punctuation-insensitive), so "Greve dos caminhoneiros..." from
two different portals is recognized as the same story.
"""

from __future__ import annotations

from urllib.parse import urlparse

from apify import Actor

from . import textmatch

# Named store that persists across runs (unlike the default per-run store).
HISTORY_STORE_NAME = 'news-history'
HISTORY_KEY = 'published-fingerprints'

# Cap how much history we keep so the record never grows unbounded.
_MAX_HISTORY = 2000

# How similar two titles must be (Jaccard of significant tokens) to be treated as the
# same story by this deterministic layer. Kept moderate on purpose: the editor AI does
# the strong *semantic* dedup; this code is the deterministic safety net for identical
# and near-identical headlines (notably the same article reappearing across runs, where
# the title is usually byte-for-byte identical and matches at 1.0). The *exact URL* match
# below is the hard guarantee that a story from a previous run is never republished.
_TITLE_SIMILARITY = 0.5


def _url_key(url: str) -> str:
    """Normalized URL identity: host + path, lowercased, without query/fragment."""
    try:
        parsed = urlparse(url or '')
    except ValueError:
        return (url or '').strip().lower()
    host = (parsed.hostname or '').lower().removeprefix('www.')
    path = (parsed.path or '').rstrip('/').lower()
    return f'{host}{path}'


def _title_tokens(title: str) -> set[str]:
    """Significant, accent-free, lowercased tokens of a title."""
    return textmatch.tokens(title)


def _similar_titles(a: set[str], b: set[str]) -> bool:
    return textmatch.jaccard(a, b) >= _TITLE_SIMILARITY


def _is_same_story(art: dict, url_keys: set[str], token_sets: list[set[str]]) -> bool:
    """True if ``art`` matches an already-seen URL (hard match) or a near-identical title."""
    if _url_key(art.get('url', '')) in url_keys:
        return True
    tokens = _title_tokens(art.get('title', ''))
    return any(_similar_titles(tokens, seen) for seen in token_sets)


def collapse_in_run(articles: list[dict]) -> list[dict]:
    """Remove articles that cover the same story, keeping the first occurrence."""
    kept: list[dict] = []
    url_keys: set[str] = set()
    token_sets: list[set[str]] = []
    removed = 0
    for art in articles:
        if _is_same_story(art, url_keys, token_sets):
            removed += 1
            continue
        kept.append(art)
        url_keys.add(_url_key(art.get('url', '')))
        token_sets.append(_title_tokens(art.get('title', '')))
    if removed:
        Actor.log.info(f'Dedup (mesma run): {removed} duplicata(s) da mesma notícia removida(s).')
    return kept


async def load_history() -> dict:
    """Load the persistent published-news history. Best-effort; never raises.

    Returns a mutable record with three views kept in sync:
    ``{"entries": list[{url_key, title_tokens}], "url_keys": set, "titles": list[set]}``.
    ``entries`` is the canonical, ordered list used for persistence; ``url_keys`` and
    ``titles`` are lookup indexes. On any failure it returns an empty record so dedup
    simply does nothing (we never block publishing because history could not be read).
    """
    record: dict = {'entries': [], 'url_keys': set(), 'titles': []}
    try:
        store = await Actor.open_key_value_store(name=HISTORY_STORE_NAME)
        raw = await store.get_value(HISTORY_KEY)
    except Exception as exc:  # noqa: BLE001 - history is best-effort
        Actor.log.warning(f'Não foi possível abrir o histórico de notícias ({exc}); seguindo sem dedup entre execuções.')
        return record

    for entry in (raw or []):
        if not isinstance(entry, dict):
            continue
        url_key = str(entry.get('url_key') or '')
        raw_tokens = entry.get('title_tokens')
        tokens = {str(t) for t in raw_tokens} if isinstance(raw_tokens, list) else set()
        record['entries'].append({'url_key': url_key, 'title_tokens': sorted(tokens)})
        if url_key:
            record['url_keys'].add(url_key)
        if tokens:
            record['titles'].append(tokens)
    Actor.log.info(f'Histórico carregado: {len(record["entries"])} notícia(s) já publicada(s) anteriormente.')
    return record


def drop_already_published(articles: list[dict], history: dict) -> list[dict]:
    """Remove articles already published in a previous run (per ``history``)."""
    url_keys = history.get('url_keys', set())
    token_sets = history.get('titles', [])
    kept: list[dict] = []
    removed = 0
    for art in articles:
        if _is_same_story(art, url_keys, token_sets):
            removed += 1
            continue
        kept.append(art)
    if removed:
        Actor.log.info(f'Dedup (entre execuções): {removed} notícia(s) já publicada(s) antes foram ignoradas.')
    return kept


def mark(history: dict, url: str, title: str) -> None:
    """Record a just-published article in the in-memory history record."""
    url_key = _url_key(url)
    tokens = _title_tokens(title)
    history.setdefault('entries', []).append({'url_key': url_key, 'title_tokens': sorted(tokens)})
    history.setdefault('url_keys', set()).add(url_key)
    if tokens:
        history.setdefault('titles', []).append(tokens)


async def save_history(history: dict) -> None:
    """Persist the updated history to the named store. Best-effort; never raises."""
    # ``entries`` is the canonical ordered list; trim to the cap, newest last.
    entries = list(history.get('entries', []))[-_MAX_HISTORY:]
    try:
        store = await Actor.open_key_value_store(name=HISTORY_STORE_NAME)
        await store.set_value(HISTORY_KEY, entries, content_type='application/json')
        Actor.log.info(f'Histórico atualizado: {len(entries)} notícia(s) registradas para não repetir.')
    except Exception as exc:  # noqa: BLE001 - history is best-effort
        Actor.log.warning(f'Não foi possível salvar o histórico de notícias ({exc}).')
