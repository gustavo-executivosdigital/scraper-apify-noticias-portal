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
    max_tokens: int | None = None,
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
        # Give long-form generations (the rewrite) enough room; without this the
        # model may stop early and the article body comes back short/truncated.
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens
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
    trusted to return FEWER than ``n`` — or an empty list — when the remaining
    articles are weak, off-topic, superficial or duplicated (a quality floor). We
    do not pad to ``n`` with junk and we do not fall back to an arbitrary article:
    publishing nothing is preferable to republishing unimportant content.
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
        'milha, infraestrutura e regulação do setor). Avalie CADA artigo INDIVIDUALMENTE e selecione '
        'apenas NOTÍCIAS REAIS E IMPORTANTES para esse público.\n\n'
        'SÓ É IMPORTANTE (score alto) conteúdo factual e noticioso, como: fatos e acontecimentos '
        'concretos, novas leis/regulamentos/normas (ANTT, ANTAQ, Receita, Anvisa, etc.), greves e '
        'paralisações, decisões de governo, mudanças de tarifas/fretes/combustível/pedágio, dados e '
        'estatísticas oficiais do setor, acordos comerciais e comércio exterior, fusões/aquisições e '
        'investimentos de empresas, obras e infraestrutura (rodovias, portos, ferrovias, aeroportos), '
        'acidentes/incidentes relevantes, e movimentos econômicos do mercado de logística.\n\n'
        'NÃO É IMPORTANTE (score baixo, NÃO selecione): conteúdo superficial ou atemporal, como listas '
        'de "dicas"/"X passos"/"X maneiras"/"melhores práticas", tutoriais e how-to, guias genéricos, '
        'colunas de opinião, publicidade/publieditorial/conteúdo patrocinado, releases promocionais de '
        'produto, matérias institucionais de autopromoção, motivacional, e qualquer coisa fora do tema '
        'de logística ou de fonte fraca/duvidosa.\n\n'
        'DEDUPLICAÇÃO: se a MESMA notícia (mesmo fato/acontecimento) aparecer em fontes diferentes ou '
        'com títulos parecidos, selecione APENAS a melhor versão (fonte mais forte e texto mais '
        'completo) e descarte TODAS as outras. Nunca selecione dois artigos que cobrem o mesmo fato.\n\n'
        'É MUITO melhor selecionar MENOS artigos (ou nenhum) do que incluir material fraco, repetido, '
        'superficial ou fora do tema. Responda SOMENTE em JSON.'
    )
    user = (
        f'Avalie individualmente os artigos abaixo e selecione no máximo {n} que sejam NOTÍCIAS REAIS '
        'E IMPORTANTES de logística para um portal. Descarte dicas/listas/tutoriais/opinião/'
        'publicidade e descarte duplicatas do mesmo fato.\n\n'
        + '\n\n'.join(catalog)
        + '\n\nResponda no formato JSON exato:\n'
        '{"selected": [{"index": <int>, "score": <0-100>, "reason": "<motivo curto em pt-BR explicando '
        'por que é uma notícia importante>"}]}\n'
        f'A lista "selected" deve ter no máximo {n} itens, ordenados do melhor para o pior. '
        'Inclua APENAS artigos com score >= 60 (notícia realmente importante); se houver menos que isso, '
        'retorne menos itens — ou uma lista vazia se nenhum prestar. NUNCA inclua duplicatas do mesmo fato '
        'nem conteúdo superficial.'
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

    # Quality floor: we trust the editor AI to return fewer than n — or NOTHING — when
    # the candidates are weak, off-topic, superficial or duplicated. We do NOT pad with
    # junk just to publish something: it is better to publish nothing this run than to
    # republish an unimportant/listicle article. An empty result is a valid, intended
    # outcome and the orchestrator handles it gracefully.
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
        'NÃO copie frases do original e NÃO adicione opinião. Escreva em português do Brasil. '
        'A matéria deve ser EXTENSA e BEM DESENVOLVIDA (texto longo de portal). Para alcançar '
        'esse tamanho, desenvolva cada ponto com profundidade jornalística: explique o contexto '
        'do setor, os antecedentes, os impactos na operação logística (transporte, frete, '
        'armazenagem, custos, prazos), as implicações para empresas e para o mercado, e as '
        'perspectivas. Você PODE ampliar com contextualização e análise setorial geral, mas é '
        'PROIBIDO inventar fatos específicos: não crie números, datas, nomes, cargos, falas, '
        'citações, estatísticas ou eventos que não estejam no texto original. A expansão deve ser '
        'contextual e analítica, nunca factual inventada. Responda SOMENTE em JSON.'
    )
    user = (
        f'{title_instruction}\n\n'
        f'Título original: {title}\n\n'
        f'Matéria original:\n{original}{truncated_note}\n\n'
        'Escreva uma matéria LONGA e completa: no mínimo 600 palavras, idealmente entre 700 e 900 '
        'palavras, com 8 a 12 parágrafos bem desenvolvidos (vários períodos cada). Não escreva '
        'parágrafos curtos nem texto raso. Você pode incluir 1 ou 2 subtítulos curtos de seção '
        '(uma linha própria, sem markdown e sem dois-pontos) para organizar o texto.\n\n'
        'Responda no formato JSON exato:\n'
        '{"title": "<novo título>", '
        '"resumo": "<resumo real da matéria em 1 a 2 frases, no máximo 280 caracteres, sem repetir o título>", '
        '"tags": ["<3 a 6 termos curtos do tema de logística, em pt-BR e minúsculas>"], '
        '"body": "<matéria reescrita, longa, 8 a 12 parágrafos separados por \\n\\n>"}'
    )

    content = await _chat(
        client, api_key, model,
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        json_mode=True, temperature=0.7, max_tokens=4096,
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
