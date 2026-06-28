"""Main entry point and orchestrator for the News Portal Rewriter Actor.

Pipeline:
1. Discover news articles on Google for the search term.
2. A Groq AI selects the best N articles from Google result metadata/snippets.
3. A Groq AI rewrites each selected article for original republication.
4. (Optional) Generate a high-quality editorial image per article (Gemini Nano Banana via OpenRouter).
5. Push one republished item per article to the dataset.

Every stage degrades gracefully: a failure in selection, rewriting, or image
generation is logged and the run continues with whatever succeeded.
"""

from __future__ import annotations

import asyncio
import inspect
import os

import httpx
from apify import Actor, Event

from . import ai_groq, dedup, discovery, image_gen


async def _kvs_public_url(store: object, key: str) -> str | None:
    """Best-effort public URL for a key in a key-value store (dict/object/SDK)."""
    getter = getattr(store, 'get_public_url', None)
    if getter is not None:
        try:
            value = getter(key)
            if inspect.isawaitable(value):
                value = await value
            if value:
                return value
        except Exception:  # noqa: BLE001 - fall back to manual construction
            pass
    store_id = getattr(store, 'id', None) or getattr(store, '_id', None)
    if store_id:
        return f'https://api.apify.com/v2/key-value-stores/{store_id}/records/{key}'
    return None


async def _maybe_generate_image(
    client: httpx.AsyncClient,
    store: object,
    *,
    index: int,
    api_key: str,
    model: str,
    title: str,
    lead: str,
) -> dict:
    """Generate, store, and return image info for one article (best-effort)."""
    prompt = image_gen.build_prompt(title, lead)
    alt = image_gen.build_alt(title)
    try:
        result = await image_gen.generate_image(
            client, api_key=api_key, model=model, prompt=prompt
        )
    except Exception as exc:  # noqa: BLE001 - image is best-effort
        Actor.log.warning(f'Image generation failed for article #{index}: {exc}')
        return {'imageUrl': None, 'imagePrompt': prompt, 'imageError': str(exc)}

    if result is None:
        return {'imageUrl': None, 'imagePrompt': prompt}

    image_bytes, content_type = result
    extension = 'png' if 'png' in content_type else 'jpg'
    key = f'image-{index}.{extension}'
    try:
        await store.set_value(key, image_bytes, content_type=content_type)
        image_url = await _kvs_public_url(store, key)
    except Exception as exc:  # noqa: BLE001
        Actor.log.warning(f'Storing image for article #{index} failed: {exc}')
        return {'imageUrl': None, 'imagePrompt': prompt, 'imageError': str(exc)}

    # ``imagemAlt`` (pt-BR spelling) is the key the portal reads for alt text; a
    # short human caption, not the long English generation prompt.
    return {'imageUrl': image_url, 'imageKey': key, 'imagePrompt': prompt, 'imagemAlt': alt}


def _confianca(score: object) -> float | None:
    """Convert a 0–100 selection score to the portal's 0–1 ``confiancaIA`` scale."""
    try:
        return round(max(0, min(100, int(float(score)))) / 100, 2)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _lead(body: str) -> str:
    """First paragraph (or first chunk) of a body, for the image prompt."""
    first = next((p.strip() for p in (body or '').split('\n') if p.strip()), '')
    return first[:300]


def _source_material(art: dict) -> str:
    """Build the best available factual source material without extra crawling."""
    body = (art.get('body') or '').strip() or (art.get('snippet') or '').strip()
    if body:
        return body
    parts = [
        f'Titulo: {art.get("title", "")}',
        f'Fonte: {art.get("source", "")}',
        f'Data: {art.get("date", "")}',
        f'URL: {art.get("url", "")}',
    ]
    return '\n'.join(p for p in parts if p.split(': ', 1)[-1].strip())


async def main() -> None:
    async with Actor:
        # Graceful abort - stop quickly when the user/platform stops the Actor.
        async def on_aborting() -> None:
            await asyncio.sleep(1)
            await Actor.exit()

        Actor.on(Event.ABORTING, on_aborting)

        # --- Input ---------------------------------------------------------------
        actor_input = await Actor.get_input() or {}
        search_query = (actor_input.get('searchQuery') or '').strip()
        max_articles = int(actor_input.get('maxArticles') or 10)
        num_to_select = int(actor_input.get('numToSelect') or 5)
        country_code = (actor_input.get('countryCode') or 'br').strip().lower()
        groq_api_key = (actor_input.get('groqApiKey') or os.environ.get('GROQ_API_KEY') or '').strip()
        groq_model = (actor_input.get('groqModel') or ai_groq.DEFAULT_MODEL).strip()
        title_style = (actor_input.get('titleStyle') or 'portal').strip().lower()
        # Date window: only consider news published within the last N days. 0 = no limit.
        last_days_raw = actor_input.get('lastDays')
        last_days = discovery.DEFAULT_LAST_DAYS if last_days_raw is None else int(last_days_raw)
        enable_image = bool(actor_input.get('enableImage', True))
        openrouter_api_key = (actor_input.get('openRouterApiKey') or os.environ.get('OPENROUTER_API_KEY') or '').strip()
        image_model = (actor_input.get('imageModel') or image_gen.DEFAULT_IMAGE_MODEL).strip()

        if not search_query:
            raise ValueError('Input "searchQuery" is required, e.g. "taxas de logística".')
        if max_articles <= 0:
            raise ValueError('Input "maxArticles" must be a positive integer.')
        if num_to_select <= 0:
            raise ValueError('Input "numToSelect" must be a positive integer.')
        if last_days < 0:
            raise ValueError('Input "lastDays" must be 0 (no limit) or a positive integer.')
        if not groq_api_key:
            raise ValueError(
                'A Groq API key is required (input "groqApiKey" or env GROQ_API_KEY) '
                'to select and rewrite the articles. Get a free key at https://console.groq.com.'
            )
        if enable_image and not openrouter_api_key:
            Actor.log.warning(
                'Image generation is ON but no OpenRouter API key was provided '
                '(input "openRouterApiKey" or env OPENROUTER_API_KEY). Skipping images; '
                'articles will still be published. Get a key at https://openrouter.ai/keys.'
            )
            enable_image = False

        # Stop before composed Apify Actors if Groq cannot publish anything.
        async with httpx.AsyncClient() as client:
            await ai_groq.preflight(client, groq_api_key, groq_model)

        # --- 1. Discover ----------------------------------------------------------
        # Funil: para entregar ``num_to_select`` notícias DISTINTAS e boas, precisamos de um
        # pool de candidatos bem maior que isso — notícias de logística costumam vir em
        # cluster (vários portais cobrindo o mesmo fato), e o dedup + filtro de importância
        # reduzem muito o conjunto. Buscamos ~4x o desejado (mantendo ``maxArticles`` como
        # piso), de forma 100% interna: a interface de entrada/saída do Actor não muda.
        candidate_pool = min(max(max_articles, num_to_select * 4), 50)
        articles = await discovery.search_news(search_query, candidate_pool, country_code, last_days)
        Actor.log.info(
            f'Funil: alvo de {num_to_select} notícia(s) distintas; buscando pool de até '
            f'{candidate_pool} candidatos.'
        )
        if not articles:
            Actor.log.warning('No news articles found for this term. Try a broader or different query.')
            return

        # --- 2. Prepare low-cost candidates ---------------------------------------
        # No composed crawler is used here. Selection and rewriting are based on the
        # Google Search result metadata/snippet plus source attribution.
        enriched: list[dict] = []
        for art in articles:
            body = _source_material(art)
            enriched.append({**art, 'body': body})

        if not enriched:
            Actor.log.warning('No usable candidates remained after discovery.')
            return
        Actor.log.info(f'{len(enriched)} articles available for low-cost selection.')

        # --- 2b. Dedup: nunca pegar notícia repetida ------------------------------
        # Collapse same-story duplicates from different sources in THIS run, then drop
        # anything we already published in a PREVIOUS run (persistent history). Both
        # steps are best-effort and never block publishing on their own failure.
        enriched = dedup.collapse_in_run(enriched)
        history = await dedup.load_history()
        enriched = dedup.drop_already_published(enriched, history)
        if not enriched:
            Actor.log.warning('Todas as notícias encontradas já foram publicadas antes ou eram duplicatas. Nada novo a publicar.')
            return

        # --- 3 + 4 + 5. Select, rewrite, illustrate -------------------------------
        store = await Actor.open_key_value_store()
        async with httpx.AsyncClient() as client:
            # Select the best N.
            try:
                picks = await ai_groq.select_best(
                    client, groq_api_key, groq_model, enriched, num_to_select
                )
            except Exception as exc:  # noqa: BLE001 - degrade: take first N in order
                Actor.log.warning(f'AI selection failed ({exc}); taking the first {num_to_select} articles.')
                picks = [{'index': i, 'score': 0, 'reason': 'Seleção por ordem (IA indisponível).'}
                         for i in range(min(num_to_select, len(enriched)))]

            Actor.log.info(f'Selected {len(picks)} articles to rewrite and publish.')

            Actor.log.info('Content crawler removed; publishing from Google Search metadata/snippets.')

            published = 0
            for position, pick in enumerate(picks):
                art = enriched[pick['index']]

                # Rewrite.
                try:
                    rewritten = await ai_groq.rewrite_article(
                        client, groq_api_key, groq_model,
                        title=art['title'], body=art['body'], title_style=title_style,
                    )
                except Exception as exc:  # noqa: BLE001 - skip this article on hard failure
                    Actor.log.warning(f'Rewrite failed for "{art["title"]}": {exc}')
                    continue
                if not rewritten['body']:
                    Actor.log.warning(f'Rewrite returned empty body for "{art["title"]}"; skipping.')
                    continue

                # Illustrate (optional).
                image_info = {'imageUrl': None, 'imagePrompt': None}
                if enable_image:
                    image_info = await _maybe_generate_image(
                        client, store,
                        index=position,
                        api_key=openrouter_api_key,
                        model=image_model,
                        title=rewritten['title'],
                        lead=_lead(rewritten['body']),
                    )

                await Actor.push_data(
                    {
                        'searchQuery': search_query,
                        'originalTitle': art['title'],
                        'originalUrl': art['url'],
                        'source': art.get('source'),
                        'publishedAt': art.get('date'),
                        'originalBody': art['body'],
                        'rewrittenTitle': rewritten['title'],
                        'rewrittenBody': rewritten['body'],
                        # Real AI summary + tags so the portal stops deriving them.
                        'resumo': rewritten.get('resumo') or None,
                        'tags': rewritten.get('tags') or [],
                        'score': pick.get('score'),
                        'selectionReason': pick.get('reason'),
                        # The portal reads ``confiancaIA`` on a 0–1 scale (it ignores
                        # ``score``, which is 0–100), so normalize it.
                        'confiancaIA': _confianca(pick.get('score')),
                        **image_info,
                    }
                )
                published += 1
                # Remember this story so future runs never republish it.
                dedup.mark(history, art['url'], art['title'])

            # Persist the updated history once, after the run (best-effort).
            if published:
                await dedup.save_history(history)

        Actor.log.info(
            f'Done. Searched "{search_query}", evaluated {len(enriched)} articles, '
            f'published {published} rewritten article(s)'
            + (' with AI images.' if enable_image else '.')
        )
        if published == 0:
            Actor.log.warning('Nothing was published. Check the Groq API key and the search term.')
