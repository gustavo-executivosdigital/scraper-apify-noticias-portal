"""News Portal Rewriter Actor.

Searches Google News for a term, extracts full article bodies, lets a Groq AI
select the best ones, rewrites each for original republication, and generates an
editorial image with Gemini 2.5 Flash Image (Nano Banana) via OpenRouter.
"""
