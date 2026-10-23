"""HTTP and SSE entry point. Run with: python -m chatbot.server"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import (ChatError, context_key, get_session_cookie, needs_search,
                   normalize_context, rate_key, system_prompt, validate_message)
from .integrations import BraveSearch, CareerApi, MemoryCache, UpstashCache
from .providers import HostedProvider


LOG = logging.getLogger("chatbot")
NO_CURRENT_FACTS = ("I couldn't verify current information right now. Please check an official "
                    "college, exam, or government source and try again later.")


@dataclass(frozen=True)
class Config:
    port: int
    allowed_origins: frozenset[str]
    career_api_base_url: str
    user_context_token: str
    redis_url: str
    redis_token: str
    hmac_secret: str
    llm_provider: str
    llm_model: str
    llm_api_key: str
    omniroute_base_url: str
    search_api_key: str

    @classmethod
    def from_env(cls) -> "Config":
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        if provider not in {"groq", "gemini", "omniroute"}:
            raise ValueError("LLM_PROVIDER must be groq, gemini, or omniroute")
        key_name = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
                    "omniroute": "OMNIROUTE_API_KEY"}[provider]
        def required(name):
            value = os.getenv(name, "").strip()
            if not value:
                raise ValueError(f"Missing {name}")
            return value
        secret = required("CACHE_HMAC_SECRET")
        if len(secret) < 32:
            raise ValueError("CACHE_HMAC_SECRET must have at least 32 characters")
        base = required("OMNIROUTE_BASE_URL") if provider == "omniroute" else ""
        redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
        redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
        if bool(redis_url) != bool(redis_token):
            raise ValueError("Set both Upstash credentials or leave both empty")
        return cls(
            port=int(os.getenv("PORT", "3001")),
            allowed_origins=frozenset(s.strip() for s in required("ALLOWED_ORIGINS").split(",") if s.strip()),
            career_api_base_url=required("CAREER_API_BASE_URL"),
            user_context_token=required("CAREER_API_USER_CONTEXT_TOKEN"),
            redis_url=redis_url,
            redis_token=redis_token,
            hmac_secret=secret, llm_provider=provider,
            llm_model=required("LLM_MODEL"), llm_api_key=required(key_name),
            omniroute_base_url=base,
            search_api_key=os.getenv("BRAVE_SEARCH_API_KEY", "").strip(),
        )


def load_dotenv() -> None:
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


class ChatService:
    def __init__(self, config: Config, *, career_api=None, cache=None, search=None, provider=None):
        self.config = config
        self.career_api = career_api or CareerApi(config.career_api_base_url, config.user_context_token)
        self.cache = cache if cache is not None else (
            UpstashCache(config.redis_url, config.redis_token)
            if config.redis_url else MemoryCache()
        )
        self.search = search if search is not None else (
            BraveSearch(config.search_api_key) if config.search_api_key else None
        )
        self.provider = provider or HostedProvider(config.llm_provider, config.llm_model,
                                                    config.llm_api_key, config.omniroute_base_url)

    def prepare(self, cookie: str, body: object):
        conversation_id, message, history = validate_message(body)
        key = context_key(self.config.hmac_secret, cookie, conversation_id)
        if self.cache.rate(rate_key(self.config.hmac_secret, cookie, time.time())) > 20:
            raise ChatError(429, "rate_limited", "Too many messages. Please wait a minute.")
        record = self.cache.get(key)
        if record is None:
            if history:
                raise ChatError(409, "conversation_expired", "This chat has expired. Start a new chat.")
            context = normalize_context(self.career_api.context(cookie))
            record = {"context": context, "created_at": time.time()}
            if not self.cache.put_first(key, record):
                record = self.cache.get(key)
                if record is None:
                    raise ChatError(503, "cache_unavailable", "Chat memory is temporarily unavailable.")
                self.career_api.check_access(cookie)
        else:
            self.career_api.check_access(cookie)
            if not self.cache.renew(key):
                raise ChatError(409, "conversation_expired", "This chat has expired. Start a new chat.")
        context = record.get("context") if isinstance(record, dict) else None
        if not isinstance(context, dict):
            raise ChatError(503, "cache_unavailable", "Chat memory is temporarily unavailable.")
        search_used = needs_search(message)
        evidence = []
        if search_used:
            if self.search is not None:
                try:
                    evidence = self.search.search(message)
                except ChatError:
                    pass
            if not evidence:
                return iter((NO_CURRENT_FACTS,)), [], search_used
        prompt = system_prompt(context, evidence)
        response = self.provider.open_stream(prompt, history, message)
        sources = [{"title": item["title"], "url": item["url"], "date": item["date"]}
                   for item in evidence]
        return self.provider.chunks(response), sources, search_used


def make_handler(service: ChatService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CareerChat/0.1"

        def log_message(self, _format, *_args):
            # BaseHTTPRequestHandler normally logs the URL. Keep logs metadata-only.
            pass

        def _json(self, status: int, body: dict):
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _event(self, name: str, data: dict):
            payload = f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": {"code": "not_found", "message": "No such endpoint."}})

        def do_POST(self):
            if self.path != "/chat/messages":
                self._json(404, {"error": {"code": "not_found", "message": "No such endpoint."}})
                return
            request_id = uuid.uuid4().hex
            started = time.monotonic()
            search_used = False
            try:
                origin = self.headers.get("Origin")
                if origin and origin not in service.config.allowed_origins:
                    raise ChatError(403, "origin_forbidden", "This origin is not allowed.")
                if self.headers.get_content_type() != "application/json":
                    raise ChatError(400, "invalid_content_type", "Send application/json.")
                cookie = get_session_cookie(self.headers.get("Cookie"))
                if not cookie:
                    raise ChatError(401, "unauthenticated", "Please log in.")
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    length = -1
                if length < 1 or length > 32768:
                    raise ChatError(400, "invalid_request", "Request body must be at most 32 KB.")
                try:
                    body = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeDecodeError):
                    raise ChatError(400, "invalid_json", "Request body is not valid JSON.") from None
                chunks, sources, search_used = service.prepare(cookie, body)
            except ChatError as error:
                self._json(error.status, {"error": {"code": error.code, "message": str(error)}})
                LOG.info("request_id=%s status=%s error=%s elapsed_ms=%d", request_id, error.status,
                         error.code, int((time.monotonic() - started) * 1000))
                return
            except Exception:
                self._json(500, {"error": {"code": "server_error", "message": "Please try again."}})
                LOG.error("request_id=%s status=500 error=server_error", request_id)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                if sources:
                    self._event("sources", {"items": sources})
                for text in chunks:
                    self._event("delta", {"text": text})
                self._event("done", {"finish_reason": "stop"})
                LOG.info("request_id=%s status=200 provider=%s search=%s elapsed_ms=%d",
                         request_id, service.config.llm_provider, search_used,
                         int((time.monotonic() - started) * 1000))
            except (BrokenPipeError, ConnectionResetError):
                close = getattr(chunks, "close", None)
                if close:
                    close()
            except ChatError as error:
                try:
                    self._event("error", {"code": error.code, "message": str(error)})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                LOG.info("request_id=%s status=stream_error error=%s", request_id, error.code)
            except Exception:
                try:
                    self._event("error", {"code": "stream_error", "message": "The reply stopped unexpectedly."})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                LOG.error("request_id=%s status=stream_error error=stream_error", request_id)

    return Handler


def main():
    load_dotenv()
    config = Config.from_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer(("0.0.0.0", config.port), make_handler(ChatService(config)))
    LOG.info("chatbot listening on port %d", config.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
