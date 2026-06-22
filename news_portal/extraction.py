"""Infallible full-text extraction via ``apify/website-content-crawler``.

The Google search snippet is never enough to rewrite an article, so we fetch the
real, cleaned body of every candidate URL. The content crawler renders pages with
a real browser (handling JavaScript-heavy news sites) and strips boilerplate,
which makes it the most reliable single source of article text.

We crawl every URL in one run (depth 0 = only the given pages), then map the
extracted text back to each article by its URL.
"""

from __future__ import annotations

from urllib.parse import urlparse

from apify import Actor

CONTENT_CRAWLER_ACTOR = 'apify/website-content-crawler'

# Below this length we assume extraction failed (cookie wall, paywall, error page).
MIN_BODY_CHARS = 250


def _canonical(url: str) -> str:
    """Normalize a URL for matching crawler output back to a discovery item."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return (url or '').strip().rstrip('/')
    host = (parsed.hostname or '').lower().removeprefix('www.')
    return f'{host}{parsed.path.rstrip("/")}'


async def fetch_bodies(urls: list[str]) -> dict[str, str]:
    """Return ``{original_url: body_text}`` for the URLs whose body we extracted.

    URLs that fail extraction (or come back too short) are simply omitted, so the
    caller naturally skips them. Raises only if the crawler returns no dataset.
    """
    clean_urls = [u for u in urls if u]
    if not clean_urls:
        return {}

    crawler_input = {
        'startUrls': [{'url': u} for u in clean_urls],
        'crawlerType': 'playwright:adaptive',
        'maxCrawlDepth': 0,
        'maxCrawlPages': len(clean_urls),
        'saveMarkdown': True,
        'saveHtml': False,
        'readableTextCharThreshold': 100,
    }
    Actor.log.info(f'Extracting full article text from {len(clean_urls)} pages via {CONTENT_CRAWLER_ACTOR}...')
    run = await Actor.call(CONTENT_CRAWLER_ACTOR, run_input=crawler_input)

    dataset_id = _dataset_id(run)
    if dataset_id is None:
        raise RuntimeError(
            f'{CONTENT_CRAWLER_ACTOR} did not return a dataset. Check the run and your account access.'
        )

    # Index extracted bodies by canonical URL so we can match them to inputs even
    # when the crawler normalizes or redirects the URL slightly.
    by_canonical: dict[str, str] = {}
    dataset = await Actor.open_dataset(id=dataset_id)
    async for item in dataset.iterate_items():
        body = (item.get('text') or item.get('markdown') or '').strip()
        if len(body) < MIN_BODY_CHARS:
            continue
        for key in {_canonical(item.get('url', '')), _canonical((item.get('crawl') or {}).get('loadedUrl', ''))}:
            if key and key not in by_canonical:
                by_canonical[key] = body

    bodies: dict[str, str] = {}
    for url in clean_urls:
        body = by_canonical.get(_canonical(url))
        if body:
            bodies[url] = body

    Actor.log.info(f'Extracted usable article text for {len(bodies)}/{len(clean_urls)} pages.')
    return bodies


def _dataset_id(run: object) -> str | None:
    if run is None:
        return None
    if isinstance(run, dict):
        return run.get('defaultDatasetId')
    return getattr(run, 'default_dataset_id', None)
