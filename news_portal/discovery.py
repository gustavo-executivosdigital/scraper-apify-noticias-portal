"""News discovery via the official ``apify/google-search-scraper`` Actor.

We ask Google for the user's term and collect the organic results, which for a
news term are overwhelmingly news articles. We return lightweight references
(title, url, source, snippet); the full article body is fetched later by
``extraction.py``.

Discovery is best-effort: any failure is raised to the caller, which logs it and
decides how to proceed.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from apify import Actor

GOOGLE_SEARCH_ACTOR = 'apify/google-search-scraper'

# Map our country code to the search Actor's allowed `languageCode` values.
# The Actor rejects bare "pt" - it only accepts "pt-BR"/"pt-PT" (and "en", etc.).
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
    return len(path) > 1


async def search_news(query: str, max_articles: int, country_code: str) -> list[dict]:
    """Return up to ``max_articles`` candidate news references for ``query``.

    Each item: ``{title, url, source, snippet}``. Raises on a hard failure of the
    search Actor so the caller can surface a clear error.
    """
    # Pull a few extra results because some will be filtered out as non-articles.
    results_per_page = min(max(max_articles * 2, 10), 100)
    search_input = {
        'queries': query,
        'resultsPerPage': results_per_page,
        'maxPagesPerQuery': 1,
        'countryCode': country_code,
        'languageCode': _LANGUAGE_BY_COUNTRY.get(country_code, 'en'),
        'mobileResults': False,
    }
    Actor.log.info(f'Searching Google news for "{query}" via {GOOGLE_SEARCH_ACTOR}...')
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

    Actor.log.info(f'Discovery found {len(articles)} candidate news articles.')
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
