"""Talking to Ollama (https://ollama.com), which runs the models on your own computer.

Only the standard library is used. The calls:
  POST /api/chat       one question, with the JSON schema the answer must follow (structured outputs);
                       Ollama's reply also says how long it took (in nanoseconds) and how many tokens
  GET  /api/tags       the models you have downloaded, with their sizes
  GET  /api/ps         the models in memory now, and how much memory they use (and how much on a GPU)
  GET  /api/version    Ollama's version, kept with the results
  POST /api/generate   with keep_alive 0: take a model out of memory, so the next one starts clean
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# The context window (tokens) for long inputs; not sent unless set, because a size different from the
# server's makes Ollama reload the model.
CONTEXT = int(os.environ.get("MODEL_CHECK_CONTEXT", "0") or 0)


class OllamaError(RuntimeError):
    """Ollama couldn't be used (not running, no such model): the run stops."""


class BadAnswer(OllamaError):
    """Ollama answered, but not with a complete JSON object: that example counts as wrong."""

    def __init__(self, message: str, timing: dict | None = None):
        super().__init__(message)
        self.timing = timing or {}


def _call(path: str, body: dict | None = None, url: str = DEFAULT_URL, timeout: float = 600) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{url.rstrip('/')}{path}", data=data,
                                     headers={"Content-Type": "application/json"})  # fmt: skip
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except TimeoutError:
        raise OllamaError(f"Ollama took longer than {timeout:.0f} seconds") from None
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]
        model = (body or {}).get("model", "")
        if e.code == 404 and "not found" in detail:
            raise OllamaError(f"Ollama doesn't have the model '{model}'. Run: ollama pull {model}") from None
        raise OllamaError(f"Ollama answered {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise OllamaError(f"Ollama took longer than {timeout:.0f} seconds") from None
        raise OllamaError(f"Can't reach Ollama at {url} ({e.reason}). Is it running? Try: ollama serve") from None


def extract_json(text: str):
    """The JSON object in a model's text: as it is, inside ```json fences, or after a few words or
    <think> notes (cloud models aren't always held to the schema). None when there is none."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for m in list(re.finditer(r"\{", text))[:20]:
        try:
            value, _ = decoder.raw_decode(text[m.start():])
            if isinstance(value, dict):
                return value
        except ValueError:
            continue
    return None


def ask(model: str, system: str, prompt: str, schema: dict, images: list[Path] = (), url: str = DEFAULT_URL,
        timeout: float = 600) -> dict:  # fmt: skip
    """One question. Returns {answer: dict, seconds, load_seconds, tokens, tokens_per_second}."""
    user = {"role": "user", "content": prompt}
    if images:
        user["images"] = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]
    body = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + [user],
        "format": schema,  # structured outputs: the answer must match this JSON schema
        "stream": False,
        "think": False,  # thinking models answer straight away; Ollama ignores it for the others
        "options": {"temperature": 0, "seed": 42,  # the same question gets the same answer: a fair, repeatable test
                    **({"num_ctx": CONTEXT} if CONTEXT else {})},
    }
    reply = _call("/api/chat", body, url, timeout)
    ns = 1e9
    total, load = reply.get("total_duration", 0) / ns, reply.get("load_duration", 0) / ns
    tokens, eval_s = reply.get("eval_count", 0), reply.get("eval_duration", 0) / ns
    timing = {"seconds": round(max(0.0, total - load), 3), "load_seconds": round(load, 3), "tokens": tokens,
              "tokens_per_second": round(tokens / eval_s, 1) if eval_s else 0.0}  # fmt: skip
    if (reply.get("done_reason") or "stop") != "stop":
        raise BadAnswer(f"the answer was cut off (done_reason: {reply.get('done_reason')})", timing)
    msg = reply.get("message") or {}
    answer = None
    for text in (msg.get("content"), msg.get("thinking")):  # cloud models may wrap it, or answer while "thinking"
        answer = extract_json(text or "")
        if isinstance(answer, dict):
            break
    if answer is None:
        raise BadAnswer(f"the answer wasn't valid JSON: {str(msg.get('content', ''))[:120]!r}", timing)
    if not isinstance(answer, dict):
        raise BadAnswer("the answer wasn't a JSON object", timing)
    return {"answer": answer, **timing}


def installed(url: str = DEFAULT_URL) -> dict[str, dict]:
    """{name: {size, parameter_size, quantization, family}} of the downloaded models."""
    out = {}
    for m in _call("/api/tags", url=url, timeout=30).get("models", []):
        d = m.get("details") or {}
        out[m["name"]] = {"size": m.get("size", 0), "parameter_size": d.get("parameter_size", ""),
                          "quantization": d.get("quantization_level", ""), "family": d.get("family", "")}  # fmt: skip
    return out


def memory(model: str, url: str = DEFAULT_URL) -> dict:
    """{size, size_vram} in bytes of a model in memory now ({} when it isn't loaded)."""
    for m in _call("/api/ps", url=url, timeout=30).get("models", []):
        if m.get("name") == model or m.get("model") == model:
            return {"size": m.get("size", 0), "size_vram": m.get("size_vram", 0)}
    return {}


def version(url: str = DEFAULT_URL) -> str:
    try:
        return _call("/api/version", url=url, timeout=30).get("version", "")
    except OllamaError:
        return ""


def unload(model: str, url: str = DEFAULT_URL) -> None:
    """Take a model out of memory (keep_alive 0), so the next model is measured on its own."""
    try:
        _call("/api/generate", {"model": model, "keep_alive": 0}, url, 60)
    except OllamaError:
        pass
