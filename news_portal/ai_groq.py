"""Groq-powered text AI: select the best articles and rewrite them.

Two responsibilities, both via Groq's OpenAI-compatible chat API:

1. ``select_best`` - rank the extracted articles and pick the strongest ``n`` for
   republication, with a score and a short reason for each.
2. ``rewrite_article`` - rewrite a selected article in fresh, original wording
   while preserving its structure and facts, so it does not read as a copy.

Every call is best-effort from the caller's perspective: errors are raised here
and the orchestrator logs them and degrades gracefully.
"""

from __future__ import annotations

import json
import re

import httpx

GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
DEFAULT_MODEL = 'llama-3.3-70b-versatile'

# If the chosen model is decommissioned/unavailable (404), fall back transparently.
FALLBACK_MODELS = ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'openai/gpt-oss-20b']

# Keep prompts bounded so we stay well inside context limits and control cost.
_MAX_BODY_CHARS_FOR_SELECTION = 1200
_MAX_BODY_CHARS_FOR_REWRITE = 8000


def _safe_json(text: str) -> dict | list | None:
    """Parse a JSON value from a model reply, tolerating prose around it."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


async def _chat(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    json_mode: bool,
    temperature: float,
) -> str:
    """Call Groq chat completions, falling back across models on a 404."""
    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_error: Exception | None = None

    for candidate in models_to_try:
        payload: dict = {
            'model': candidate,
            'messages': messages,
            'temperature': temperature,
        }
        if json_mode:
            payload['response_format'] = {'type': 'json_object'}
        try:
            response = await client.post(
                GROQ_URL,
                headers={'Authorization': f'Bearer {api_key}'},
                json=payload,
                timeout=120,
            )
        except httpx.HTTPError as exc:
            last_error = exc
            continue

        if response.status_code == 404:
            last_error = RuntimeError(f'Groq model "{candidate}" not available (404).')
            continue
        if response.status_code != 200:
            raise RuntimeError(f'Groq API error {response.status_code}: {response.text[:300]}')

        data = response.json()
        return data['choices'][0]['message']['content']

    raise RuntimeError(f'All Groq models failed. Last error: {last_error}')


async def select_best(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    articles: list[dict],
    n: int,
) -> list[dict]:
    """Pick up to ``n`` best articles. Returns each as ``{index, score, reason}``.

    ``index`` is the position in the input ``articles`` list. The editor AI is
    trusted to return FEWER than ``n`` when the remaining articles are weak or
    off-topic (a quality floor) — we do not pad to ``n`` with junk. Only if the AI
    returns nothing usable do we fall back to the first article, so a run still
    has something to publish.
    """
    catalog = []
    for i, art in enumerate(articles):
        body_preview = (art.get('body') or '')[:_MAX_BODY_CHARS_FOR_SELECTION]
        catalog.append(
            f'### Artigo {i}\n'
            f'Título: {art.get("title", "")}\n'
            f'Fonte: {art.get("source", "")}\n'
            f'Trecho: {body_preview}'
        )

    system = (
        'Você é o editor-chefe de um portal brasileiro ESPECIALIZADO EM LOGÍSTICA (transporte, '
        'frete, cadeia de suprimentos, comércio exterior, rodovias/portos, armazenagem, última '
        'milha, infraestrutura e regulação do setor). Avalie os artigos e selecione apenas os que '
        'realmente interessam a esse público, priorizando relevância para o setor de logística, '
        'atualidade, clareza e credibilidade da fonte. '
        'Penalize fortemente (score baixo) o que fugir do tema de logística, for opinião/publicidade '
        'ou de fonte fraca. NÃO selecione a mesma notícia repetida em fontes diferentes — escolha a '
        'melhor versão e descarte as duplicatas. É melhor selecionar MENOS artigos do que incluir '
        'material fraco ou fora do tema. Responda SOMENTE em JSON.'
    )
    user = (
        f'Selecione até {n} dos MELHORES artigos da lista abaixo para um portal de logística.\n\n'
        + '\n\n'.join(catalog)
        + '\n\nResponda no formato JSON exato:\n'
        '{"selected": [{"index": <int>, "score": <0-100>, "reason": "<motivo curto em pt-BR>"}]}\n'
        f'A lista "selected" deve ter no máximo {n} itens, ordenados do melhor para o pior. '
        'Inclua apenas artigos com score >= 50; se houver menos que isso, retorne menos itens. '
        'Não inclua duplicatas da mesma notícia.'
    )

    content = await _chat(
        client, api_key, model,
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        json_mode=True, temperature=0.2,
    )
    parsed = _safe_json(content) or {}
    raw = parsed.get('selected', []) if isinstance(parsed, dict) else []

    picks: list[dict] = []
    used: set[int] = set()
    for item in raw:
        try:
            idx = int(item.get('index'))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(articles) or idx in used:
            continue
        used.add(idx)
        picks.append(
            {
                'index': idx,
                'score': _coerce_score(item.get('score')),
                'reason': str(item.get('reason') or '').strip(),
            }
        )
        if len(picks) >= n:
            break

    # Quality floor: we trust the editor AI to return fewer than n when the rest is
    # weak or off-topic — we do NOT pad with junk just to reach n. Only as a last
    # resort, if the AI returned nothing usable, fall back to the first article so
    # the run still produces something to publish.
    if not picks and articles:
        picks.append({'index': 0, 'score': 0, 'reason': 'Seleção por ordem (IA não retornou candidatos).'})

    return picks


async def rewrite_article(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    *,
    title: str,
    body: str,
    title_style: str,
) -> dict:
    """Rewrite an article for original republication. Returns ``{title, body}``."""
    title_instruction = (
        'Crie um título novo, chamativo e jornalístico (estilo portal), sem clickbait enganoso.'
        if title_style == 'portal'
        else 'Mantenha o título fiel ao sentido do original, apenas com palavras próprias.'
    )

    original = body[:_MAX_BODY_CHARS_FOR_REWRITE]
    truncated_note = (
        '\n\n(OBS: o texto acima pode estar truncado; reescreva apenas com base no que recebeu, '
        'sem inventar um final.)'
        if len(body) > _MAX_BODY_CHARS_FOR_REWRITE
        else ''
    )

    system = (
        'Você é um jornalista de um portal brasileiro especializado em LOGÍSTICA, transporte, '
        'frete, cadeia de suprimentos, comércio exterior, rodovias/portos e regulação do setor. '
        'Reescreva a matéria para republicação com palavras 100% próprias, preservando TODOS os '
        'fatos, nomes, números e a estrutura jornalística (lide + corpo em parágrafos). Use o '
        'vocabulário do setor de logística quando couber, mantendo o tom informativo e neutro. '
        'NÃO copie frases do original, NÃO invente fatos, NÃO adicione opinião. Escreva em '
        'português do Brasil. Responda SOMENTE em JSON.'
    )
    user = (
        f'{title_instruction}\n\n'
        f'Título original: {title}\n\n'
        f'Matéria original:\n{original}{truncated_note}\n\n'
        'Responda no formato JSON exato:\n'
        '{"title": "<novo título>", '
        '"resumo": "<resumo real da matéria em 1 a 2 frases, no máximo 280 caracteres, sem repetir o título>", '
        '"tags": ["<3 a 6 termos curtos do tema de logística, em pt-BR e minúsculas>"], '
        '"body": "<matéria reescrita, 3 a 6 parágrafos separados por \\n\\n>"}'
    )

    content = await _chat(
        client, api_key, model,
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        json_mode=True, temperature=0.7,
    )
    parsed = _safe_json(content) or {}
    if not isinstance(parsed, dict):
        parsed = {}

    raw_tags = parsed.get('tags')
    tags: list[str] = []
    if isinstance(raw_tags, list):
        seen: set[str] = set()
        for tag in raw_tags:
            clean = str(tag or '').strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                tags.append(clean)
            if len(tags) >= 6:
                break

    return {
        'title': str(parsed.get('title') or title).strip(),
        'resumo': str(parsed.get('resumo') or '').strip()[:280],
        'tags': tags,
        'body': str(parsed.get('body') or '').strip(),
    }


def _coerce_score(value: object) -> int:
    try:
        score = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))
