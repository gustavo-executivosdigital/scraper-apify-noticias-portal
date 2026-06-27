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

import asyncio
import json
import re

import httpx
from apify import Actor

from . import textmatch

# Free Groq tiers cap tokens-per-minute (e.g. 12k TPM). A heavy rewrite (~8-9k tokens)
# means the next call in the same minute hits HTTP 429. We wait it out and retry instead
# of failing: honor the server's Retry-After, else wait a full minute (the TPM window).
_MAX_RATE_LIMIT_RETRIES = 6
_DEFAULT_RATE_LIMIT_WAIT = 60.0

GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
DEFAULT_MODEL = 'llama-3.3-70b-versatile'

# If the chosen model is decommissioned/unavailable (404), fall back transparently.
FALLBACK_MODELS = ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'openai/gpt-oss-20b']

# Keep prompts bounded so we stay well inside context limits and control cost.
# Give the selector enough body to judge IMPORTANCE (not just a snippet), but stay
# bounded for cost. The rewrite gets the most context.
_MAX_BODY_CHARS_FOR_SELECTION = 2200
_MAX_BODY_CHARS_FOR_REWRITE = 8000

# Total budget of article-preview characters in ONE selection prompt. The free Groq tier
# caps tokens-per-minute (~12k), and a single request bigger than that can NEVER succeed
# (retrying doesn't help). ~28k chars ≈ 7k tokens keeps the whole selection call safely
# under the cap even with a large candidate pool. Per-article preview shrinks to fit.
_SELECTION_TOTAL_CHAR_BUDGET = 28000
_MIN_BODY_CHARS_FOR_SELECTION = 400

# Hard quality floor enforced IN CODE (not just requested in the prompt): any article
# the editor AI scored below this is dropped even if it returned it. The prompt asks
# for >= 60; we keep the code floor slightly lower (50) so we respect the model's own
# scoring without fighting minor miscalibration, while still blocking anything the
# model itself flagged as weak. The goal is "only real, important news" — never junk.
_MIN_SELECTION_SCORE = 50

# Title patterns that strongly signal superficial/listicle/how-to content (the kind the
# user explicitly does NOT want). We do NOT auto-drop on these (a headline like "Governo
# anuncia 5 medidas para os portos" is real news) — we only FLAG them in the catalog so
# the editor AI penalizes them. The AI makes the final call.
_SUPERFICIAL_TITLE_RE = re.compile(
    r'(\b\d+\s+(dicas?|passos?|maneiras?|formas?|motivos?|raz[õo]es|truques?|segredos?|'
    r'erros?|mitos?|coisas?)\b)'
    r'|(\bpasso\s+a\s+passo\b)'
    r'|(\bcomo\s+(fazer|escolher|melhorar|reduzir|economizar|aumentar|montar|criar)\b)'
    r'|(\b(guia|tutorial)\s+(completo|definitivo|pr[áa]tico|de)\b)'
    r'|(\bmelhores\s+(dicas|pr[áa]ticas|formas|maneiras)\b)',
    re.IGNORECASE,
)


async def preflight(client: httpx.AsyncClient, api_key: str, model: str) -> None:
    """Validate the Groq key/model with a tiny request before expensive discovery.

    This intentionally does not use the retry-heavy ``_chat`` helper: if Groq is
    restricted, invalid, out of credit, or rate-limited, we want to stop before
    calling composed Apify Actors.
    """
    payload = {
        'model': model or DEFAULT_MODEL,
        'messages': [{'role': 'user', 'content': 'OK'}],
        'temperature': 0,
        'max_tokens': 1,
    }
    try:
        response = await client.post(
            GROQ_URL,
            headers={'Authorization': f'Bearer {api_key}'},
            json=payload,
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f'Groq preflight failed: {exc}') from exc

    if response.status_code == 200:
        return
    raise RuntimeError(f'Groq preflight failed {response.status_code}: {response.text[:300]}')


def _looks_superficial(title: str) -> bool:
    """Heuristic flag for listicle/how-to/tips titles (soft signal, not an auto-drop)."""
    return bool(_SUPERFICIAL_TITLE_RE.search(title or ''))


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
    """Call Groq chat completions, retrying on rate limits and falling back on 404."""
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

        # Retry THIS model on 429 (tokens-per-minute limit). We wait the time the server
        # asks for (or a full minute) and try again, so free-tier runs simply pace
        # themselves instead of failing the article.
        rate_limited = False
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = await client.post(
                    GROQ_URL,
                    headers={'Authorization': f'Bearer {api_key}'},
                    json=payload,
                    timeout=180,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                break  # network error: try the next model

            if response.status_code == 429:
                last_error = RuntimeError('Groq rate limit (429).')
                if attempt >= _MAX_RATE_LIMIT_RETRIES:
                    rate_limited = True
                    break
                wait = _retry_after_seconds(response) or _DEFAULT_RATE_LIMIT_WAIT
                Actor.log.warning(
                    f'Groq atingiu o limite de tokens/minuto (429). Aguardando {wait:.0f}s e '
                    f'tentando de novo (tentativa {attempt + 1}/{_MAX_RATE_LIMIT_RETRIES})...'
                )
                await asyncio.sleep(wait)
                continue

            if response.status_code == 404:
                last_error = RuntimeError(f'Groq model "{candidate}" not available (404).')
                break  # try the next model
            if response.status_code != 200:
                raise RuntimeError(f'Groq API error {response.status_code}: {response.text[:300]}')

            data = response.json()
            return data['choices'][0]['message']['content']

        # If we exhausted rate-limit retries, switching model won't help (same account
        # TPM budget), so stop and surface the error.
        if rate_limited:
            break

    raise RuntimeError(f'All Groq models failed. Last error: {last_error}')


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Seconds to wait before retrying, parsed from Groq's rate-limit headers.

    Honors ``Retry-After`` (seconds) and the ``x-ratelimit-reset-*`` headers (values
    like ``"6.5s"`` or ``"1m30s"``). Adds a 1s safety buffer. Returns ``None`` if no
    usable hint is present, letting the caller fall back to a default wait.
    """
    for header in ('retry-after', 'x-ratelimit-reset-tokens', 'x-ratelimit-reset-requests'):
        value = response.headers.get(header)
        if not value:
            continue
        seconds = _parse_duration(value)
        if seconds is not None:
            return seconds + 1.0
    return None


def _parse_duration(value: str) -> float | None:
    """Parse ``"45"``, ``"6.5s"``, ``"2m"``, ``"1m30s"``, ``"850ms"`` into seconds."""
    text = (value or '').strip().lower()
    if not text:
        return None
    # Plain number = seconds.
    try:
        return float(text)
    except ValueError:
        pass
    if text.endswith('ms') and text[:-2].replace('.', '', 1).isdigit():
        return float(text[:-2]) / 1000
    total = 0.0
    matched = False
    for amount, unit in re.findall(r'([\d.]+)\s*(ms|s|m|h)', text):
        matched = True
        num = float(amount)
        total += num * {'ms': 0.001, 's': 1, 'm': 60, 'h': 3600}[unit]
    return total if matched else None


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
    # Size each article's preview so the whole selection prompt fits the free-tier
    # per-minute token budget, even with a large candidate pool.
    per_article_chars = _MAX_BODY_CHARS_FOR_SELECTION
    if articles:
        per_article_chars = max(
            _MIN_BODY_CHARS_FOR_SELECTION,
            min(_MAX_BODY_CHARS_FOR_SELECTION, _SELECTION_TOTAL_CHAR_BUDGET // len(articles)),
        )

    catalog = []
    for i, art in enumerate(articles):
        body_preview = (art.get('body') or '')[:per_article_chars]
        title = art.get('title', '')
        date = art.get('date') or 'não informada'
        flag = (
            '\n⚠ ALERTA: o título parece conteúdo superficial (lista/dica/tutorial/how-to). '
            'Só selecione se for, de fato, uma notícia real e importante; caso contrário dê score baixo.'
            if _looks_superficial(title)
            else ''
        )
        catalog.append(
            f'### Artigo {i}\n'
            f'Título: {title}\n'
            f'Fonte: {art.get("source", "")}\n'
            f'Data: {date}\n'
            f'Trecho: {body_preview}'
            f'{flag}'
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
        'AGRUPAR POR ASSUNTO (passo OBRIGATÓRIO antes de selecionar): muitos artigos cobrem o MESMO '
        'acontecimento/assunto com palavras diferentes — por exemplo "Amazon investe em centro de '
        'distribuição", "Amazon expande no Brasil" e "Amazon inaugura novo centro" são TODOS o MESMO '
        'assunto. Agrupe mentalmente os artigos por assunto/evento real (empresa + acontecimento) e, '
        'de cada grupo, selecione APENAS UM — a melhor versão (fonte mais forte, texto mais completo e '
        'recente). É TERMINANTEMENTE PROIBIDO selecionar dois artigos do mesmo assunto/evento, mesmo '
        'que o enfoque, o título ou as palavras sejam diferentes. Cada item selecionado deve tratar de '
        'um assunto DISTINTO dos demais.\n\n'
        'É MUITO melhor selecionar MENOS artigos (ou nenhum) do que incluir material fraco, repetido, '
        'superficial ou fora do tema. Prefira variedade de assuntos distintos e relevantes. '
        'Responda SOMENTE em JSON.'
    )
    user = (
        f'Avalie individualmente os artigos abaixo e selecione no máximo {n} que sejam NOTÍCIAS REAIS '
        'E IMPORTANTES de logística para um portal, cada uma sobre um ASSUNTO DISTINTO. Descarte '
        'dicas/listas/tutoriais/opinião/publicidade e, quando vários artigos forem do mesmo assunto/'
        'evento, escolha só a melhor versão e descarte as demais.\n\n'
        + '\n\n'.join(catalog)
        + '\n\nResponda no formato JSON exato:\n'
        '{"selected": [{"index": <int>, "score": <0-100>, '
        '"assunto": "<rótulo curto e canônico do assunto/evento, 2 a 5 palavras, ex.: \\"amazon centro '
        'distribuicao\\" ou \\"antt reajuste pedagio\\">", '
        '"reason": "<motivo curto em pt-BR explicando por que é uma notícia importante>"}]}\n'
        f'A lista "selected" deve ter no máximo {n} itens, todos de ASSUNTOS DISTINTOS, ordenados do '
        'melhor para o pior. Inclua APENAS artigos com score >= 60 (notícia realmente importante); se '
        'houver menos assuntos distintos que isso, retorne menos itens — ou uma lista vazia se nenhum '
        'prestar. NUNCA inclua dois itens do mesmo assunto/evento nem conteúdo superficial.'
    )

    content = await _chat(
        client, api_key, model,
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        json_mode=True, temperature=0.2,
    )
    parsed = _safe_json(content) or {}
    raw = parsed.get('selected', []) if isinstance(parsed, dict) else []

    # First pass: validate index, enforce the hard score floor, capture each pick's
    # subject label. We sort by score so the per-subject collapse keeps the strongest.
    candidates: list[dict] = []
    used: set[int] = set()
    for item in raw:
        try:
            idx = int(item.get('index'))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(articles) or idx in used:
            continue
        score = _coerce_score(item.get('score'))
        # Hard quality floor: never publish what the AI itself scored as weak.
        if score < _MIN_SELECTION_SCORE:
            continue
        used.add(idx)
        # Keep the AI's canonical subject label and the real title as SEPARATE token sets.
        # The label compared against other labels is the strongest same-subject signal;
        # mixing it with the noisy title would dilute it. Title tokens are a fallback for
        # when the AI omits/weakens the label.
        assunto = str(item.get('assunto') or '').strip()
        title_tokens = textmatch.tokens(articles[idx].get('title', ''))
        subject_tokens = textmatch.tokens(assunto) or title_tokens
        candidates.append(
            {
                'index': idx,
                'score': score,
                'reason': str(item.get('reason') or '').strip(),
                '_subject': subject_tokens,
                '_title': title_tokens,
            }
        )

    candidates.sort(key=lambda c: c['score'], reverse=True)

    # Second pass — deterministic SEMANTIC dedup backstop: even if the AI slipped and
    # returned two picks about the same subject (e.g. three "Amazon novo centro"
    # articles), keep only the highest-scored one per subject. We collapse when the
    # canonical subject labels strongly overlap OR the titles are near-identical. This
    # is what guarantees the published set never has near-duplicate stories.
    picks: list[dict] = []
    kept: list[dict] = []
    for cand in candidates:
        duplicate = any(
            textmatch.overlap_coefficient(cand['_subject'], k['_subject']) >= 0.6
            or textmatch.jaccard(cand['_title'], k['_title']) >= 0.5
            for k in kept
        )
        if duplicate:
            Actor.log.info(
                f'Dedup semântico (seleção): artigo #{cand["index"]} descartado por ser do mesmo '
                f'assunto de outro já selecionado.'
            )
            continue
        kept.append(cand)
        picks.append({k: v for k, v in cand.items() if not k.startswith('_')})
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
    short_source = len((body or '').strip()) < 500
    truncated_note = (
        '\n\n(OBS: o texto acima pode estar truncado; reescreva apenas com base no que recebeu, '
        'sem inventar um final.)'
        if len(body) > _MAX_BODY_CHARS_FOR_REWRITE
        else ''
    )
    if short_source:
        length_instruction = (
            'A fonte abaixo e curta. Escreva uma nota factual curta, direta e publicavel, '
            'com 2 a 4 paragrafos, usando apenas as informacoes disponiveis. Nao invente '
            'contexto, numeros, impacto, declaracoes ou desdobramentos que nao estejam no material.'
        )
    else:
        length_instruction = (
            'Escreva uma materia substancial e bem desenvolvida, com pelo menos 1000 caracteres. '
            'Quando a materia original for rica em informacao, va mais longe e aproveite ao maximo '
            'todos os fatos disponiveis. Quando a fonte for pobre, desenvolva o que houver com '
            'clareza, sem inventar nem repetir ideias so para alongar. Paragrafos separados por \\n\\n.'
        )

    system = (
        'Você é um jornalista de um portal brasileiro especializado em LOGÍSTICA, transporte, '
        'frete, cadeia de suprimentos, comércio exterior, rodovias/portos e regulação do setor. '
        'Reescreva a matéria para republicação com palavras 100% próprias, preservando TODOS os '
        'fatos, nomes, números, datas e a estrutura jornalística (lide + corpo em parágrafos). Use o '
        'vocabulário do setor de logística quando couber, mantendo o tom informativo e neutro. '
        'NÃO copie frases do original e NÃO adicione opinião. Escreva em português do Brasil.\n\n'
        'REGRA DE OURO DO CONTEÚDO: o texto deve ser construído com a INFORMAÇÃO REAL da matéria '
        'original — fatos, dados, números, nomes, declarações e contexto que REALMENTE estão na fonte. '
        'EXTRAIA o máximo de informação concreta do original e desenvolva cada fato com clareza e '
        'profundidade jornalística (o quê, quem, quando, onde, por quê, e o impacto na operação '
        'logística: transporte, frete, armazenagem, custos, prazos). NÃO ENCHA LINGUIÇA: é proibido '
        'preencher com frases genéricas, vagas, repetitivas ou "de contexto" só para aumentar o '
        'tamanho, e é proibido INVENTAR qualquer número, data, nome, cargo, fala, estatística ou '
        'evento que não esteja na fonte. Prefira densidade de informação real a volume vazio. '
        'Responda SOMENTE em JSON.'
    )
    user = (
        f'{title_instruction}\n\n'
        f'Título original: {title}\n\n'
        f'Matéria original:\n{original}{truncated_note}\n\n'
        'Escreva uma matéria substancial e bem desenvolvida, com NO MÍNIMO ~1000 caracteres. Quando a '
        'matéria original for rica em informação, vá MAIS LONGE e aproveite ao máximo todos os fatos '
        'disponíveis (faça o texto crescer com substância real, não com enchimento). Quando a fonte '
        'for pobre, desenvolva o que houver com clareza, sem inventar nem repetir ideias só para '
        'alongar. Parágrafos separados por \\n\\n; você pode usar 1 ou 2 subtítulos curtos de seção '
        '(uma linha própria, sem markdown e sem dois-pontos) quando o conteúdo justificar.\n\n'
        + (f'\n\nREGRA FINAL PARA FONTE CURTA: {length_instruction}\n\n' if short_source else '')
        + 'Responda no formato JSON exato:\n'
        '{"title": "<novo título>", '
        '"resumo": "<resumo real da matéria em 1 a 2 frases, no máximo 280 caracteres, sem repetir o título>", '
        '"tags": ["<3 a 6 termos curtos do tema de logística, em pt-BR e minúsculas>"], '
        '"body": "<matéria reescrita, com substância real, parágrafos separados por \\n\\n>"}'
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
