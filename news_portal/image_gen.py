"""Pluggable AI image generation for the news image.

Two providers behind one interface so the rest of the Actor never cares which is
used:

- ``gemini``  - Google Gemini 2.5 Flash Image ("Nano Banana"). High quality;
                requires a Google AI Studio API key. This is the default.
- ``pollinations`` - free, no API key. Useful as a zero-cost fallback for testing
                or when no Gemini key is available.

``generate_image`` returns the raw PNG/JPEG bytes (or ``None`` if generation was
skipped), and raises on a hard provider error so the orchestrator can log and
degrade without crashing the run. Adding a new provider = one new ``_gen_*``
function plus a branch here.
"""

from __future__ import annotations

import base64
from urllib.parse import quote

import httpx

GEMINI_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
POLLINATIONS_ENDPOINT = 'https://image.pollinations.ai/prompt/{prompt}'


def build_prompt(title: str, lead: str) -> str:
    """Build an editorial image prompt from a news title and lead paragraph."""
    summary = f'{title}. {lead}'.strip()
    summary = summary[:600]
    return (
        'Professional editorial news photograph illustrating this story: '
        f'"{summary}". Photorealistic, high quality, natural lighting, journalistic style, '
        '16:9 composition. No text, no captions, no watermark, no logos.'
    )


async def generate_image(
    client: httpx.AsyncClient,
    *,
    provider: str,
    api_key: str,
    model: str,
    prompt: str,
) -> tuple[bytes, str] | None:
    """Generate an image. Returns ``(image_bytes, content_type)`` or ``None``.

    Raises ``RuntimeError`` / ``httpx.HTTPError`` on provider failure.
    """
    if provider == 'gemini':
        return await _gen_gemini(client, api_key, model, prompt)
    if provider == 'pollinations':
        return await _gen_pollinations(client, prompt)
    raise RuntimeError(f'Unknown image provider "{provider}".')


async def _gen_gemini(
    client: httpx.AsyncClient, api_key: str, model: str, prompt: str
) -> tuple[bytes, str]:
    """Call Gemini 2.5 Flash Image and return the first inline image."""
    if not api_key:
        raise RuntimeError('Gemini API key is required for the "gemini" image provider.')

    url = GEMINI_ENDPOINT.format(model=model)
    payload = {'contents': [{'parts': [{'text': prompt}]}]}
    response = await client.post(
        url,
        params={'key': api_key},
        json=payload,
        timeout=180,
    )
    if response.status_code != 200:
        raise RuntimeError(f'Gemini API error {response.status_code}: {response.text[:300]}')

    data = response.json()
    for candidate in data.get('candidates', []):
        for part in (candidate.get('content') or {}).get('parts', []):
            inline = part.get('inlineData') or part.get('inline_data')
            if inline and inline.get('data'):
                content_type = inline.get('mimeType') or inline.get('mime_type') or 'image/png'
                return base64.b64decode(inline['data']), content_type

    raise RuntimeError('Gemini returned no image data (the model may have refused the prompt).')


async def _gen_pollinations(client: httpx.AsyncClient, prompt: str) -> tuple[bytes, str]:
    """Call the free, keyless Pollinations image endpoint."""
    url = POLLINATIONS_ENDPOINT.format(prompt=quote(prompt))
    response = await client.get(
        url,
        params={'width': 1280, 'height': 720, 'nologo': 'true', 'safe': 'true'},
        timeout=180,
        follow_redirects=True,
    )
    if response.status_code != 200:
        raise RuntimeError(f'Pollinations error {response.status_code}: {response.text[:200]}')
    content_type = response.headers.get('content-type', 'image/jpeg').split(';')[0]
    return response.content, content_type
