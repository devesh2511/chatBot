"""Streaming adapters for Groq, Gemini, and OmniRoute."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .core import ChatError


def sse_data(response):
    """Yield each SSE data payload, including frames with multiple data lines."""
    lines = []
    for raw in response:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip())
    if lines:
        yield "\n".join(lines)


class HostedProvider:
    def __init__(self, provider: str, model: str, api_key: str, base_url: str = ""):
        if provider not in {"groq", "gemini", "omniroute"}:
            raise ValueError("LLM_PROVIDER must be groq, gemini, or omniroute")
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def open_stream(self, prompt: str, history: list[dict], message: str):
        if self.provider == "gemini":
            url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{quote(self.model, safe='')}:streamGenerateContent?alt=sse")
            contents = [{"role": "model" if turn["role"] == "assistant" else "user",
                         "parts": [{"text": turn["content"]}]} for turn in history]
            contents.append({"role": "user", "parts": [{"text": message}]})
            body = {"systemInstruction": {"parts": [{"text": prompt}]},
                    "contents": contents,
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1200}}
            headers = {"x-goog-api-key": self.api_key}
        else:
            url = ("https://api.groq.com/openai/v1/chat/completions" if self.provider == "groq"
                   else f"{self.base_url}/chat/completions")
            body = {"model": self.model, "stream": True, "temperature": 0.3,
                    "max_tokens": 1200,
                    "messages": [{"role": "system", "content": prompt}, *history,
                                 {"role": "user", "content": message}]}
            headers = {"Authorization": f"Bearer {self.api_key}"}
        request = Request(url, data=json.dumps(body).encode(), method="POST",
                          headers={"Content-Type": "application/json", **headers})
        try:
            return urlopen(request, timeout=45)
        except (HTTPError, URLError, TimeoutError) as error:
            raise ChatError(502, "model_unavailable", "The model is unavailable. Please retry.") from error

    def chunks(self, response):
        emitted = False
        try:
            for data in sse_data(response):
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if event.get("error"):
                    raise ChatError(502, "model_unavailable", "The model stopped unexpectedly.")
                if self.provider == "gemini":
                    candidates = event.get("candidates") or []
                    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
                    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
                else:
                    choices = event.get("choices") or []
                    text = choices[0].get("delta", {}).get("content", "") if choices else ""
                if isinstance(text, str) and text:
                    emitted = True
                    yield text
            if not emitted:
                raise ChatError(502, "empty_model_response", "The model did not return an answer.")
        except (OSError, TimeoutError) as error:
            raise ChatError(502, "model_unavailable", "The model stream stopped unexpectedly.") from error
        finally:
            response.close()
