"""AI image generation for the news image, via OpenRouter.

Uses OpenRouter's OpenAI-compatible chat API with Google Gemini 2.5 Flash Image
("Nano Banana", model ``google/gemini-2.5-flash-image``). The image is requested
by setting ``modalities: ["image", "text"]`` and comes back as a base64 data URL
in ``choices[0].message.images[0].image_url.url``.

``generate_image`` returns the raw image bytes (or ``None`` if skipped) and raises
on a hard provider error so the orchestrator can log and degrade without crashing
the run.
"""

from __future__ import annotations

import base64
import binascii
import re

import httpx

OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
DEFAULT_IMAGE_MODEL = 'google/gemini-2.5-flash-image'

_DATA_URL_RE = re.compile(r'^data:(?P<mime>[^;,]+)?(?:;base64)?,(?P<data>.+)$', re.DOTALL)


def build_prompt(title: str, lead: str) -> str:
    """Build a hyper-realistic, news-coherent image prompt that contains NO text.

    The prompt describes an authentic, un-staged photojournalistic image that
    matches the story. It is engineered to look like a real photograph shot by a
    press photographer — NOT like an AI render — by specifying real camera gear,
    natural light, documentary framing and physical imperfections, and by
    explicitly banning the "AI look" (plastic skin, over-smoothing, HDR glow,
    3D/CGI/illustration). It also forbids any letters, words, captions, logos or
    watermarks so the image is clean and reusable.
    """
    summary = ' '.join(f'{title}. {lead}'.split())[:600]
    return (
        'A real, authentic press photograph documenting this Brazilian news story: '
        f'"{summary}". '
        'It must look like a genuine photo captured on assignment by a professional '
        'photojournalist for a newspaper — a real moment, candid and un-staged, NOT a '
        'concept render. '
        'This is for a Brazilian LOGISTICS news portal, so when the story allows, ground the '
        'scene in the real logistics world of Brazil: diesel trucks on a BR highway, port and '
        'container terminals, cargo ships, distribution warehouses, forklifts and pallets, '
        'loading docks, supply-chain and freight operations, with believable Brazilian '
        'environment, vehicles and workwear. '
        'Technical look: shot on a full-frame DSLR (Canon EOS R5 / Nikon) with a 35mm or 50mm '
        'prime lens, natural available light, true-to-life and slightly imperfect exposure, '
        'realistic shallow depth of field with natural background blur (bokeh), authentic color '
        'grading (not over-saturated), fine natural sensor grain, real-world textures — worn '
        'metal, dust, scratches, asphalt, weathered surfaces and fabric. '
        '35mm reportage / documentary photography aesthetic, 16:9 horizontal composition, '
        'suitable as the lead image of a news article. '
        'CRITICAL — it must NOT look AI-generated: no plastic or waxy skin, no airbrushed or '
        'over-smoothed surfaces, no glossy CGI sheen, no exaggerated HDR glow, no unrealistic '
        'symmetry, no over-perfect lighting, no 3D render, no illustration, no digital painting, '
        'no cartoon, no concept art. Keep human faces, hands and proportions anatomically '
        'correct and natural; no distorted hands, no warped or deformed objects, no surreal '
        'artifacts, no extra limbs or fingers. '
        'Absolutely NO text of any kind in the image: no words, no letters, no numbers, '
        'no captions, no headlines, no signs with readable text, no logos, no watermarks, '
        'no UI overlays. Do not depict real identifiable public figures. '
        'The image must be tasteful, neutral, and safe for a general news audience.'
    )


def build_alt(title: str) -> str:
    """Short, human pt-BR alt text for the generated image (SEO + accessibility).

    The image is generated to illustrate the story, so a concise descriptive alt
    derived from the headline is accurate and far better than dumping the long
    English generation prompt into the ``alt`` attribute.
    """
    clean = ' '.join((title or '').split()).strip().rstrip('.')
    if not clean:
        return 'Imagem ilustrativa de logística'
    return f'Imagem ilustrativa: {clean}'[:160]


async def generate_image(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    model: str,
    prompt: str,
) -> tuple[bytes, str] | None:
    """Generate an image via OpenRouter. Returns ``(image_bytes, content_type)``.

    Raises ``RuntimeError`` / ``httpx.HTTPError`` on provider failure.
    """
    if not api_key:
        raise RuntimeError('OpenRouter API key is required to generate images.')

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'modalities': ['image', 'text'],
    }
    response = await client.post(
        OPENROUTER_URL,
        headers={
            'Authorization': f'Bearer {api_key}',
            'HTTP-Referer': 'https://apify.com',
            'X-Title': 'Noticias Portal Rewriter',
        },
        json=payload,
        timeout=180,
    )
    if response.status_code != 200:
        raise RuntimeError(f'OpenRouter API error {response.status_code}: {response.text[:300]}')

    data = response.json()
    choices = data.get('choices') or []
    if not choices:
        raise RuntimeError(f'OpenRouter returned no choices: {str(data)[:200]}')

    message = choices[0].get('message') or {}
    for image in (message.get('images') or []):
        url = ((image.get('image_url') or {}).get('url')) or ''
        decoded = _decode_data_url(url)
        if decoded:
            return decoded

    raise RuntimeError('OpenRouter returned no image data (the model may have refused the prompt).')


def _decode_data_url(url: str) -> tuple[bytes, str] | None:
    """Decode a ``data:<mime>;base64,<data>`` URL into ``(bytes, content_type)``."""
    match = _DATA_URL_RE.match(url or '')
    if not match:
        return None
    content_type = match.group('mime') or 'image/png'
    try:
        return base64.b64decode(match.group('data')), content_type
    except (binascii.Error, ValueError):
        return None
