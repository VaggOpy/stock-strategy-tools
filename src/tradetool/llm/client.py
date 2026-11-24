"""Two LLM backends with the same chat() call, and a disk cache in front of them.

The cache matters for the free Gemini tier (a few hundred requests a day at best): an experiment
that is run again reads the saved answers instead of asking again, and gives the same result.
"""
from __future__ import annotations

import hashlib
import json
import re
import time

import requests

from tradetool.config import DATA_DIR, env

CACHE_DIR = DATA_DIR / "llm_cache"


class LLMError(RuntimeError):
    pass


class Gemini:
    def __init__(self, model: str | None = None, key: str | None = None):
        key = key or env("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set in .env (free key: https://aistudio.google.com/apikey)")
        from google import genai

        self.model = model or env("GEMINI_MODEL") or "gemini-flash-lite-latest"
        self.name = f"gemini/{self.model}"
        self.version = None  # the exact model behind an alias like *-latest, from the last answer
        self._client = genai.Client(api_key=key)

    def chat(self, system: str, messages: list[dict], as_json: bool = False) -> str:
        from google.genai import errors, types

        contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                  parts=[types.Part(text=m["text"])]) for m in messages]
        config = types.GenerateContentConfig(system_instruction=system, temperature=0,
                                             response_mime_type="application/json" if as_json else None)
        for attempt in range(4):
            try:
                r = self._client.models.generate_content(model=self.model, contents=contents, config=config)
                self.version = r.model_version
                return r.text or ""
            except errors.ServerError as e:  # 503 when the model is overloaded, usually short
                if attempt == 3:
                    raise LLMError(f"Gemini server error {e.code}: {e.message}") from e
                time.sleep(5 * 2 ** attempt)
            except errors.ClientError as e:
                wait = _retry_delay(e) if e.code == 429 else None
                if wait is None or wait > 70 or attempt == 3:
                    if e.code == 429:
                        raise LLMError(f"Gemini quota used up for {self.model} (the free tier has a daily "
                                       "limit per project). Try later or use another model.") from e
                    raise LLMError(f"Gemini error {e.code}: {e.message}") from e
                time.sleep(wait + 1)  # per-minute limit, wait as told
        raise LLMError("unreachable")


def _retry_delay(e) -> float | None:
    """Seconds to wait from a 429 answer, None if it is a daily limit."""
    text = json.dumps(getattr(e, "details", None) or getattr(e, "response_json", None) or {})
    if "PerDay" in text:
        return None
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', text)
    return float(m.group(1)) if m else 30.0


class Ollama:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or env("OLLAMA_MODEL") or "qwen2.5:3b"
        self.host = host or env("OLLAMA_HOST") or "http://localhost:11434"
        self.name = f"ollama/{self.model}"
        self.version = self.model

    def chat(self, system: str, messages: list[dict], as_json: bool = False) -> str:
        body = {"model": self.model, "stream": False, "options": {"temperature": 0, "seed": 0},
                "messages": [{"role": "system", "content": system}]
                + [{"role": m["role"], "content": m["text"]} for m in messages]}
        if as_json:
            body["format"] = "json"
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=600)
        except requests.ConnectionError as e:
            raise LLMError(f"Ollama is not running on {self.host}") from e
        if r.status_code != 200:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:200]}")
        return r.json()["message"]["content"]


class Cached:
    """Same chat() as the wrapped client. Answers are saved as one JSON file per question."""

    def __init__(self, client, folder=CACHE_DIR):
        self.client = client
        self.name = client.name
        self.folder = folder
        self.hits = 0
        self.calls = 0

    def chat(self, system: str, messages: list[dict], as_json: bool = False) -> str:
        key = hashlib.sha256(json.dumps([self.name, system, messages, as_json]).encode()).hexdigest()[:32]
        path = self.folder / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text = self.client.chat(system, messages, as_json)
        self.calls += 1
        self.folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.name, "version": self.client.version, "time": time.time(),
                                    "messages": messages, "text": text}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        return text


def default_client(cache: bool = True):
    """Gemini if there is a key, else Ollama if it answers, else LLMError."""
    if env("GEMINI_API_KEY"):
        client = Gemini()
    else:
        client = Ollama()
        try:
            requests.get(f"{client.host}/api/tags", timeout=3)
        except requests.ConnectionError:
            raise LLMError("No LLM available: put GEMINI_API_KEY in .env "
                           "(free key: https://aistudio.google.com/apikey) or start Ollama.") from None
    return Cached(client) if cache else client
