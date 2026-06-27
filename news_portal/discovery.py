"""News discovery.

Primary source: the ``fabri-lab/apify-google-news-scraper`` Actor, which queries
**Google News** (not generic web search). Google News is news-native: results come
ranked by relevance and can be restricted by recency (``timePeriod``), so we get the
most relevant *recent* coverage of the term — exactly what a news portal wants. With
``extractFullText`` the Actor also returns the article body, which we carry through so
the rewrite step has real content even when our own extractor can't reach the page.

Fallback: if the Google News Actor fails or returns nothing, we transparently fall
back to the original ``apify/google-search-scraper`` (generic Google organic results),
so a run never breaks because of the discovery source.

Both paths return the same lightweight reference shape:
``{title, url, source, snippet, date, body?}``.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from apify import Actor

# Primary: Google News (relevance + recency, pt-BR/BR aware, optional full text).
GOOGLE_NEWS_ACTOR = 'fabri-lab/apify-google-news-scraper'
# Fallback: generic Google organic search (the original source).
GOOGLE_SEARCH_ACTOR = 'apify/google-search-scraper'

# Default recency window, in days. Only news published within the last N days are
# considered, so we surface fresh, relevant stories. 0 = no date limit (any time).
DEFAULT_LAST_DAYS = 2

# Map our country code to the Google News Actor's (googleCountry, uiLanguage). Its enums
# have no Portugal option, so "pt" uses Brazilian context as the closest match.
_NEWS_LOCALE = {'br': ('BR', 'pt'), 'pt': ('BR', 'pt'), 'us': ('US', 'en')}

# Map our country code to the (fallback) search Actor's allowed `languageCode` values.
# That Actor rejects bare "pt" - it only accepts "pt-BR"/"pt-PT" (and "en", etc.).
_LANGUAGE_BY_COUNTRY = {'br': 'pt-BR', 'pt': 'pt-PT', 'us': 'en'}

# Domains that are not actual news articles - we skip them as candidates (fallback path).
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
    r'apartamento.*loca[cç][aã]o|imove[lil]|operador-logistico',
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


def _google_search_fallback_enabled() -> bool:
    """Cost guard: generic Google Search fallback is opt-in via env, not default."""
    value = os.environ.get('NEWS_ENABLE_GOOGLE_SEARCH_FALLBACK', '')
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _google_news_full_text_enabled() -> bool:
    """Cost guard: Google News full-text extraction is opt-in via env."""
    value = os.environ.get('NEWS_EXTRACT_FULL_TEXT_IN_GOOGLE_NEWS', '')
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


async def search_news(
    query: str,
    max_articles: int,
    country_code: str,
    last_days: int = DEFAULT_LAST_DAYS,
) -> list[dict]:
    """Return up to ``max_articles`` candidate news references for ``query``.

    Only news published within the last ``last_days`` days are considered (``0`` = no
    date limit). Tries Google News first (relevance-ranked, date-filtered); on any
    failure or empty result, falls back to generic Google organic search. Each item:
    ``{title, url, source, snippet, date, body?}``.
    """
    window = 'sem limite de data' if last_days <= 0 else f'últimos {last_days} dia(s)'
    try:
        articles = await _search_google_news(query, max_articles, country_code, last_days)
        if articles:
            Actor.log.info(
                f'Discovery (Google News, {window}) encontrou {len(articles)} '
                f'candidato(s) ordenados por relevância.'
            )
            return articles
        if not _google_search_fallback_enabled():
            Actor.log.warning(
                'Google News nao retornou candidatos; fallback de busca organica '
                'desligado para evitar custo com resultados genericos.'
            )
            return []
        Actor.log.warning('Google News não retornou candidatos; usando fallback (busca orgânica).')
    except Exception as exc:  # noqa: BLE001 - degrade to the original source
        if not _google_search_fallback_enabled():
            Actor.log.warning(
                f'Actor de Google News falhou ({exc}); fallback de busca organica '
                'desligado para evitar custo com resultados genericos.'
            )
            return []
        Actor.log.warning(f'Actor de Google News falhou ({exc}); usando fallback (busca orgânica).')

    return await _search_google_organic(query, max_articles, country_code, last_days)


def _date_window(last_days: int) -> tuple[str, str] | None:
    """Return ``(min_date, max_date)`` ISO strings for the last ``last_days`` days.

    ``None`` when there is no date limit (``last_days <= 0``).
    """
    if last_days <= 0:
        return None
    today = datetime.now(timezone.utc).date()
    return (today - timedelta(days=last_days)).isoformat(), today.isoformat()


async def _search_google_news(
    query: str, max_articles: int, country_code: str, last_days: int
) -> list[dict]:
    """Discover via the Google News Actor: relevance-ranked, date-filtered."""
    google_country, ui_language = _NEWS_LOCALE.get(country_code, ('US', 'en'))
    run_input = {
        'searchQuery': query,
        'googleCountry': google_country,
        'uiLanguage': ui_language,
        'maxItems': min(max(max_articles, 1), 1000),
        # Pull the article body too, so the rewrite has real content regardless of
        # whether our own extractor can reach the (often redirected) publisher URL.
        'extractFullText': _google_news_full_text_enabled(),
        # Let Google drop near-duplicate / omitted results.
        'filter': True,
    }
    # Date filter: the Actor's `timePeriod` only takes fixed enums, so for an arbitrary
    # "last N days" we use a custom range (min = N days ago, max = today). 0 = any time.
    window = _date_window(last_days)
    if window is None:
        run_input['timePeriod'] = 'all'
    else:
        run_input['timePeriod'] = 'custom'
        run_input['customTimePeriodMin'], run_input['customTimePeriodMax'] = window
    Actor.log.info(
        f'Buscando notícias de "{query}" via {GOOGLE_NEWS_ACTOR} '
        f'(país={google_country}, idioma={ui_language}, '
        f'janela={"qualquer data" if window is None else f"{window[0]}..{window[1]}"})...'
    )
    run = await Actor.call(GOOGLE_NEWS_ACTOR, run_input=run_input)

    dataset_id = _dataset_id(run)
    if dataset_id is None:
        raise RuntimeError(f'{GOOGLE_NEWS_ACTOR} não retornou um dataset.')

    seen: set[str] = set()
    articles: list[dict] = []
    dataset = await Actor.open_dataset(id=dataset_id)
    async for item in dataset.iterate_items():
        url = (item.get('link') or item.get('url') or '').strip()
        title = (item.get('title') or '').strip()
        if not url or not url.startswith('http') or not title or url in seen:
            continue
        seen.add(url)
        full_text = (item.get('fullText') or '').strip()
        article = {
            'title': title,
            'url': url,
            'source': (item.get('source') or _host(url)).strip(),
            'snippet': (item.get('snippet') or item.get('description') or '').strip(),
            'date': (item.get('date') or item.get('publishedAt') or '').strip() or None,
        }
        # Carry the body only when the Actor actually extracted it; the orchestrator
        # prefers it and only falls back to its own extractor when it is missing.
        if full_text:
            article['body'] = full_text
        articles.append(article)
        if len(articles) >= max_articles:
            break

    return articles


async def _search_google_organic(
    query: str, max_articles: int, country_code: str, last_days: int
) -> list[dict]:
    """Fallback: the original generic Google organic search via the search Actor."""
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
    Actor.log.info(f'Fallback: buscando "{query}" via {GOOGLE_SEARCH_ACTOR}...')
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

    Actor.log.info(f'Discovery (fallback) found {len(articles)} candidate news articles.')
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
