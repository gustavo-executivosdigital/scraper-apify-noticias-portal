Searches **Google News** for any term, picks the strongest stories with AI, rewrites each one as an **original article** ready to republish, and generates an **editorial image** for it — all in one run.

## What does Notícias Portal Rewriter do?

Give it a search term like `taxas de logística` or `zé felipe e ana maria`. The Actor finds recent news on Google, extracts the **full text** of each article, then uses a **Groq AI** to select the best ones (you choose how many). Each selected story is **rewritten in fresh, original wording** — same facts and structure, no copied sentences — so it is safe to republish on your own portal. Finally it generates a matching **news image** with **Google Gemini 2.5 Flash Image** (or a free, keyless fallback).

Running on the Apify platform gives you API access, scheduling, integrations, proxy rotation, and run monitoring out of the box.

## Why use Notícias Portal Rewriter?

- **Content portals** that republish news at scale without manual copywriting.
- **Newsletters and social pages** that need fresh, original takes on trending stories.
- **SEO** — original text plus an original image avoids duplicate-content penalties.
- **Speed** — discovery, selection, rewriting, and illustration in a single automated run.

## How to use Notícias Portal Rewriter

1. Enter a **search term** (`searchQuery`).
2. Set **how many articles to fetch** (`maxArticles`) and **how many the AI should select** (`numToSelect`).
3. Paste your **Groq API key** (free at https://console.groq.com).
4. Paste your **Gemini API key** (free at https://aistudio.google.com/apikey) to generate high-quality images. Without it, the Actor still publishes the rewritten text, just without images.
5. Click **Start** and read the results in the **Output** tab.

## Input

| Field | Description | Default |
| --- | --- | --- |
| `searchQuery` | Term to search on Google News. **Required.** | — |
| `maxArticles` | How many articles to fetch and extract. | `10` |
| `numToSelect` | How many of the best the AI selects to rewrite. | `5` |
| `countryCode` | Country for the Google search (`br`, `pt`, `us`). | `br` |
| `groqApiKey` | Groq API key for selection + rewriting. **Required.** | — |
| `groqModel` | Groq chat model. | `llama-3.3-70b-versatile` |
| `titleStyle` | `portal` (catchy) or `faithful`. | `portal` |
| `enableImage` | Generate an AI image per article. | `true` |
| `geminiApiKey` | Google AI Studio key (required for images). | — |
| `geminiImageModel` | Gemini image model. | `gemini-2.5-flash-image` |

## Output

Each republished article is one dataset item. You can download the dataset in JSON, HTML, CSV, or Excel. Generated images are stored in the run's key-value store and linked from `imageUrl`.

```json
{
  "searchQuery": "taxas de logística",
  "originalTitle": "Governo estuda nova taxa para o setor de transportes",
  "originalUrl": "https://exemplo.com.br/noticia",
  "source": "exemplo.com.br",
  "publishedAt": "2 days ago",
  "rewrittenTitle": "Setor de transportes pode ter nova taxa; entenda o impacto",
  "rewrittenBody": "O governo avalia a criação de uma nova cobrança...\n\n...",
  "score": 92,
  "selectionReason": "Tema atual e de alto interesse para o público de logística.",
  "imageUrl": "https://api.apify.com/v2/key-value-stores/.../records/image-0.png",
  "imagePrompt": "Professional editorial news photograph illustrating..."
}
```

## Data table

| Field | Meaning |
| --- | --- |
| `rewrittenTitle` / `rewrittenBody` | The original, republishable article. |
| `originalTitle` / `originalUrl` / `source` | Where the story came from (attribution). |
| `score` / `selectionReason` | Why the AI picked this story. |
| `imageUrl` / `imagePrompt` | The generated image and the prompt used. |

## Cost estimation

Cost comes mainly from two composed Actors — `apify/google-search-scraper` (discovery) and `apify/website-content-crawler` (full-text extraction) — plus this Actor's compute. The AI calls run on **your own free Groq and Gemini keys**, so the text and image generation add no platform cost. Fetch fewer articles (`maxArticles`) to reduce Compute Units.

## Tips

- Lower `maxArticles` and raise `numToSelect` close to it to spend less on extraction.
- Set `enableImage: false` to skip image generation (and its Gemini quota) when you only need the rewritten text.
- Schedule the Actor (e.g. hourly) to keep a portal continuously fed with fresh stories.

## FAQ, disclaimers, and support

- **Is republishing legal?** You are responsible for how you use the output. Rewriting preserves facts but you should still credit sources and respect each publisher's Terms of Service and copyright.
- **Why was a story skipped?** Some sites block extraction (paywalls, cookie walls); those articles are dropped before selection.
- **Image didn't generate?** Check your Gemini key/quota — the Actor logs the reason and still publishes the text. Feedback and issues are welcome in the Issues tab.
