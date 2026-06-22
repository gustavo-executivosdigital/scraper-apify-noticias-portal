"""AI image generation for the news image, via Google Gemini.

Uses Google Gemini 2.5 Flash Image ("Nano Banana"), which needs a Google AI
Studio API key. ``generate_image`` returns the raw image bytes (or ``None`` if
skipped) and raises on a hard provider error so the orchestrator can log and
degrade without crashing the run.
"""

from __future__ import annotations

import base64

import httpx

GEMINI_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'


def build_prompt(title: str, lead: str) -> str:
    """Build a high-quality, news-coherent image prompt that contains NO text.

    The prompt describes a realistic editorial photograph that matches the story,
    and explicitly forbids any letters, words, captions, logos, or watermarks so
    the generated image is clean and reusable.
    """
    summary = ' '.join(f'{title}. {lead}'.split())[:600]
    return (
        'Create a high-quality, photorealistic editorial news photograph that '
        f'accurately and coherently illustrates this news story: "{summary}". '
        'Style: professional photojournalism, realistic natural lighting, sharp focus, '
        'rich detail, depth of field, 16:9 horizontal composition, suitable as the '
        'lead image of a news article. '
        'Absolutely NO text of any kind in the image: no words, no letters, no numbers, '
        'no captions, no headlines, no signs with readable text, no logos, no watermarks, '
        'no UI overlays. Do not depict real identifiable public figures. '
        'The image must be tasteful, neutral, and safe for a general news audience.'
    )


async def generate_image(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    model: str,
    prompt: str,
) -> tuple[bytes, str] | None:
    """Generate an image with Gemini. Returns ``(image_bytes, content_type)``.

    Raises ``RuntimeError`` / ``httpx.HTTPError`` on provider failure.
    """
    if not api_key:
        raise RuntimeError('Gemini API key is required to generate images.')

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
