"""LLM providers.

- OpenAICompatibleLLM: one client for any OpenAI-compatible /chat/completions endpoint
  (Ollama, Groq, OpenAI, Gemini, OpenRouter, vLLM, LM Studio ...). Swap providers via .env only.
- OfflineExtractiveLLM: no network, no key. Picks the best-supported sentences from the sources and
  cites them. Keeps demos, tests and CI free and deterministic.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .text import split_sentences, tokenize

ABSTAIN_MESSAGE = "I don't know based on the documents you have access to."


@dataclass
class LLMResult:
    text: str
    usage: dict[str, int] = field(default_factory=dict)


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str, sources: list[str], question: str) -> LLMResult: ...


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 600,
        timeout_s: float = 60.0,
    ):
        self.name = f"openai-compatible:{model}"
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_s

    def generate(self, system: str, user: str, sources: list[str], question: str) -> LLMResult:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        resp = httpx.post(self._url, json=payload, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        return LLMResult(
            text=data["choices"][0]["message"]["content"].strip(),
            usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
        )


class OfflineExtractiveLLM:
    """Extractive answerer: scores each source sentence by IDF-weighted overlap with the question."""

    name = "offline-extractive"

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def generate(self, system: str, user: str, sources: list[str], question: str) -> LLMResult:
        q = set(tokenize(question))
        if not q or not sources:
            return LLMResult(ABSTAIN_MESSAGE)
        sentences = [(i + 1, s) for i, src in enumerate(sources) for s in split_sentences(src)]
        df: dict[str, int] = {}
        for _, s in sentences:
            for t in set(tokenize(s)):
                df[t] = df.get(t, 0) + 1
        n = len(sentences)

        def score(sentence: str, src_rank: int) -> float:
            toks = set(tokenize(sentence))
            pref = {t[:5] for t in toks if len(t) > 5}
            hit = sum(
                math.log(1 + n / (1 + df.get(t, 0))) * (1.0 if t in toks else 0.6)
                for t in q
                if t in toks or (len(t) > 5 and t[:5] in pref)
            )
            has_number = 0.15 if re.search(r"\d", sentence) else 0.0
            return hit * (1.0 / (1 + 0.1 * (src_rank - 1))) + has_number

        ranked = sorted(((score(s, i), i, s) for i, s in sentences), reverse=True)
        top = ranked[0][0] if ranked else 0.0
        # keep supporting sentences only if they are nearly as relevant as the best one
        best = [r for r in ranked if r[0] > 0 and r[0] >= 0.6 * top][: self.max_sentences]
        if not best:
            return LLMResult(ABSTAIN_MESSAGE)
        # keep the original reading order of the chosen sentences
        best.sort(key=lambda r: (r[1], sentences.index((r[1], r[2]))))
        return LLMResult(" ".join(f"{s.rstrip()} [{i}]" for _, i, s in best))


def build_llm(settings) -> LLM:
    if settings.llm_provider == "openai":
        return OpenAICompatibleLLM(
            settings.llm_base_url,
            settings.llm_model,
            settings.llm_api_key,
            settings.llm_temperature,
            settings.llm_max_tokens,
            settings.llm_timeout_s,
        )
    return OfflineExtractiveLLM()
