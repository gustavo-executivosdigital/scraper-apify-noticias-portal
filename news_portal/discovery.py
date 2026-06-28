"""News discovery through Google Search organic results only.

This module intentionally composes a single external Apify Actor:
``apify/google-search-scraper``. Keeping discovery to one Actor makes fanout and
run cost predictable.

Returned items use the lightweight reference shape:
``{title, url, source, snippet, date}``.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from apify import Actor

GOOGLE_SEARCH_ACTOR = 'apify/google-search-scraper'

# Default recency window, in days. Only news published within the last N days are
# considered, so we surface fresh stories. 0 = no date limit (any time).
DEFAULT_LAST_DAYS = 2

# Map our country code to the search Actor's allowed `languageCode` values.
# That Actor rejects bare "pt" - it only accepts "pt-BR"/"pt-PT" (and "en", etc.).
_LANGUAGE_BY_COUNTRY = {'br': 'pt-BR', 'pt': 'pt-PT', 'us': 'en'}

# Domains that are not actual news articles - we skip them as candidates.
_SKIP_HOSTS = (
    'google.com',
    'youtube.com',
    'facebook.com',
    'instagram.com',
    'twitter.com',
    'x.com',
    'tiktok.com',
    'wikipedia.org',
    'linkedin.com',
    'infojobs.com.br',
    'adzuna.com.br',
    'jobleads.com',
    'talent.com',
    'net-empregos.com',
    'jobfy.pro',
    'indeed.com',
    'catho.com.br',
    'vagas.com.br',
    'glassdoor.com',
    'bne.com.br',
    'olx.com.br',
    'zapimoveis.com.br',
    'vivareal.com.br',
    'quintoandar.com.br',
    'imovelweb.com.br',
    'lopestiete.com.br',
    'bassanesi.com.br',
    'exonimoveis.com.br',
)

_SKIP_PATH_RE = re.compile(
    r'/(?:jobs?|vagas?|empregos?)(?:/|$)|vaga-de|detalhes-do-imovel|'
    r'apartamento.*loca|imove[lil]|operador-logistico',
    re.IGNORECASE,
)


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or '').lower().removeprefix('www.')
    except ValueError:
        return ''


def _is_article(url: str) -> bool:
    """Heuristic: a real article URL, not a homepage, aggregator, or social page."""
    if not url or not url.startswith('http'):
        return False
    host = _host(url)
    if not host or any(host == h or host.endswith('.' + h) for h in _SKIP_HOSTS):
        return False
    # An article usually has a path (slug), not just the domain root.
    path = urlparse(url).path.strip('/')
    if len(path) <= 1:
        return False
    return not _SKIP_PATH_RE.search('/' + path)


async def search_news(
    query: str,
    max_articles: int,
    country_code: str,
    last_days: int = DEFAULT_LAST_DAYS,
) -> list[dict]:
    """Return up to ``max_articles`` candidate news references for ``query``.

    Only Google Search organic results are used. Recency is constrained with the
    Google ``after:YYYY-MM-DD`` operator when ``last_days`` is positive.
    """
    return await _search_google_organic(query, max_articles, country_code, last_days)


def _date_window(last_days: int) -> tuple[str, str] | None:
    """Return ``(min_date, max_date)`` ISO strings for the last ``last_days`` days.

    ``None`` when there is no date limit (``last_days <= 0``).
    """
    if last_days <= 0:
        return None
    today = datetime.now(timezone.utc).date()
    return (today - timedelta(days=last_days)).isoformat(), today.isoformat()


async def _search_google_organic(
    query: str, max_articles: int, country_code: str, last_days: int
) -> list[dict]:
    """Discover articles via the Google Search Results Scraper Actor."""
    # Pull a few extra results because some will be filtered out as non-articles.
    results_per_page = min(max(max_articles * 2, 10), 100)
    # Restrict to recent results with Google's `after:` date operator (0 = no limit).
    window = _date_window(last_days)
    effective_query = query if window is None else f'{query} after:{window[0]}'
    search_input = {
        'queries': effective_query,
        'resultsPerPage': results_per_page,
        'maxPagesPerQuery': 1,
        'countryCode': country_code,
        'languageCode': _LANGUAGE_BY_COUNTRY.get(country_code, 'en'),
        'mobileResults': False,
    }
    Actor.log.info(
        f'Buscando candidatos de "{query}" via {GOOGLE_SEARCH_ACTOR} '
        f'(janela={"qualquer data" if window is None else f"{window[0]}..{window[1]}"})...'
    )
    run = await Actor.call(GOOGLE_SEARCH_ACTOR, run_input=search_input)

    dataset_id = _dataset_id(run)
    if dataset_id is None:
        raise RuntimeError(
            f'{GOOGLE_SEARCH_ACTOR} did not return a dataset. Check the run and your account access.'
        )

    seen: set[str] = set()
    articles: list[dict] = []
    dataset = await Actor.open_dataset(id=dataset_id)
    async for page in dataset.iterate_items():
        for result in (page.get('organicResults') or []):
            url = (result.get('url') or '').strip()
            if not _is_article(url) or url in seen:
                continue
            seen.add(url)
            articles.append(
                {
                    'title': (result.get('title') or '').strip(),
                    'url': url,
                    'source': _source_name(result, url),
                    'snippet': (result.get('description') or '').strip(),
                    'date': (result.get('date') or '').strip() or None,
                }
            )
            if len(articles) >= max_articles:
                break
        if len(articles) >= max_articles:
            break

    Actor.log.info(f'Discovery (Google Search) found {len(articles)} candidate news articles.')
    return articles


def _source_name(result: dict, url: str) -> str:
    """Best-effort publisher/source name for a search result."""
    for field in ('displayedUrl', 'source'):
        value = result.get(field)
        if value:
            cleaned = re.sub(r'^https?://(?:www\.)?', '', str(value)).split('/')[0]
            if cleaned:
                return cleaned
    return _host(url)


def _dataset_id(run: object) -> str | None:
    """Read the default dataset id from an Actor run (tolerates dict or object)."""
    if run is None:
        return None
    if isinstance(run, dict):
        return run.get('defaultDatasetId')
    return getattr(run, 'default_dataset_id', None)
