import importlib
import sys
import types
import unittest


class _FakeActor:
    log = types.SimpleNamespace(info=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None)


_FakeEvent = types.SimpleNamespace(ABORTING="ABORTING")


sys.modules.setdefault("apify", types.SimpleNamespace(Actor=_FakeActor, Event=_FakeEvent))
sys.modules.setdefault("httpx", types.SimpleNamespace(HTTPError=Exception, AsyncClient=object, Response=object))

discovery = importlib.import_module("news_portal.discovery")
ai_groq = importlib.import_module("news_portal.ai_groq")
main_module = importlib.import_module("news_portal.main")


class DiscoveryCostControlsTest(unittest.TestCase):
    def test_discovery_uses_only_google_search_actor(self):
        self.assertEqual(discovery.GOOGLE_SEARCH_ACTOR, "apify/google-search-scraper")
        self.assertFalse(hasattr(discovery, "GOOGLE_NEWS_ACTOR"))
        self.assertFalse(hasattr(discovery, "_search_google_news"))
        self.assertFalse(hasattr(discovery, "_google_news_full_text_enabled"))
        self.assertFalse(hasattr(discovery, "_google_search_fallback_enabled"))

    def test_content_crawler_runtime_hook_is_removed(self):
        self.assertFalse(hasattr(main_module, "_content_crawler_enabled"))
        self.assertFalse(hasattr(main_module, "extraction"))

    def test_article_filter_rejects_job_and_classified_urls(self):
        bad_urls = [
            "https://www.infojobs.com.br/vaga-de-operador-logistico.aspx",
            "https://br.linkedin.com/jobs/view/operador-logistico",
            "https://www.adzuna.com.br/santos/operador-de-monitoramento",
            "https://www.jobleads.com/br/job/operador-de-logistica-i-junior",
            "https://www.lopestiete.com.br/detalhes-do-imovel/?Galpao/Deposito/Armazem",
        ]

        for url in bad_urls:
            with self.subTest(url=url):
                self.assertFalse(discovery._is_article(url))


class _FakeResponse:
    def __init__(self, status_code, text="ok", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {"choices": [{"message": {"content": "OK"}}]}
        self.headers = {}

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class GroqPreflightTest(unittest.IsolatedAsyncioTestCase):
    async def test_preflight_uses_tiny_completion_before_expensive_discovery(self):
        client = _FakeClient(_FakeResponse(200))

        await ai_groq.preflight(client, "key", "llama-3.3-70b-versatile")

        self.assertEqual(len(client.calls), 1)
        payload = client.calls[0][1]["json"]
        self.assertEqual(payload["max_tokens"], 1)
        self.assertEqual(payload["temperature"], 0)

    async def test_preflight_raises_on_restricted_organization(self):
        client = _FakeClient(
            _FakeResponse(
                400,
                '{"error":{"message":"Organization has been restricted","code":"organization_restricted"}}',
            )
        )

        with self.assertRaisesRegex(RuntimeError, "Organization has been restricted"):
            await ai_groq.preflight(client, "key", "llama-3.3-70b-versatile")


class RewritePromptCostControlsTest(unittest.IsolatedAsyncioTestCase):
    async def test_short_source_material_asks_for_concise_factual_note(self):
        client = _FakeClient(
            _FakeResponse(
                200,
                payload={"choices": [{"message": {"content": '{"title":"t","resumo":"r","tags":[],"body":"b"}'}}]},
            )
        )

        await ai_groq.rewrite_article(
            client,
            "key",
            "llama-3.3-70b-versatile",
            title="ANTT publica nova regra para frete",
            body="Titulo: ANTT publica nova regra para frete\nFonte: Exemplo",
            title_style="portal",
        )

        messages = client.calls[0][1]["json"]["messages"]
        joined = "\n".join(m["content"] for m in messages)
        self.assertIn("REGRA FINAL PARA FONTE CURTA", joined)
        self.assertIn("nota factual curta", joined)


if __name__ == "__main__":
    unittest.main()
