"""Running a task: every example through every model, each answer scored and saved as it comes.

Results go to <task>/results/<model>.jsonl, one line per example, so a run that is stopped (Ctrl+C,
a power cut) carries on where it was. Each model also gets <model>.meta.json: its size, the memory it
used, Ollama's version and the task's fingerprint. When task.toml or examples.jsonl change, the
fingerprint changes, and the old results are set aside instead of being mixed with new ones.
"""

import datetime as dt
import json
import re
import time
from pathlib import Path

from . import ollama, scoring
from . import task as tasks


def safe(model: str) -> str:
    """A file name for a model: "qwen3:4b" -> "qwen3_4b"."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def results_dir(task: dict) -> Path:
    return Path(task["folder"]) / "results"


def load_results(task: dict, model: str) -> list[dict]:
    path = results_dir(task) / f"{safe(model)}.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("fingerprint") == task["fingerprint"]]


def load_meta(task: dict, model: str) -> dict:
    path = results_dir(task) / f"{safe(model)}.meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def models_with_results(task: dict) -> list[str]:
    """The models with results for this version of the task, in the order they were first run."""
    found = []
    for meta in sorted(results_dir(task).glob("*.meta.json"), key=lambda p: p.stat().st_mtime):
        m = json.loads(meta.read_text())
        if m.get("fingerprint") == task["fingerprint"] and load_results(task, m["model"]):
            found.append(m["model"])
    return found


def run_model(task: dict, model: str, ask=None, url: str = ollama.DEFAULT_URL, again: bool = False,
              limit: int = 0, say=print) -> list[dict]:  # fmt: skip
    """All examples (or the first `limit`) through one model; examples already answered are skipped."""
    ask = ask or (lambda **kw: ollama.ask(url=url, **kw))
    folder = results_dir(task)
    folder.mkdir(exist_ok=True)
    path = folder / f"{safe(model)}.jsonl"
    if again and path.exists():
        path.unlink()
    elif path.exists() and not load_results(task, model):  # results of an older version of the task
        old = path.with_name(f"{path.stem}.before-{dt.date.today().isoformat()}.jsonl")
        path.rename(old)
        say(f"{model}: the task changed since its last run; the old results are kept in {old.name}")
    done = {r["id"] for r in load_results(task, model)}
    todo = [e for e in task["examples"][: limit or None] if e["id"] not in done]
    schema = tasks.schema(task)
    first_load = 0.0
    for n, ex in enumerate(todo, 1):
        started = time.monotonic()
        row = {"id": ex["id"], "model": model, "fingerprint": task["fingerprint"], "answer": None, "error": ""}
        try:
            got = ask(model=model, system=task["system"], prompt=tasks.prompt(task, ex), schema=schema,
                      images=[Path(task["folder"]) / p for p in ex.get("images") or []])  # fmt: skip
            row["answer"] = got["answer"]
            timing = got
        except ollama.BadAnswer as e:  # a broken answer is a wrong answer, not a reason to stop
            row["error"], timing = str(e), e.timing
        row.update({k: timing.get(k, 0) for k in ("seconds", "load_seconds", "tokens", "tokens_per_second")})
        if not row["seconds"]:  # no timing from Ollama (an old version): the clock on the wall
            row["seconds"] = round(time.monotonic() - started, 3)
        first_load = first_load or row["load_seconds"]
        row.update(scoring.score(task, row["answer"], ex["expected"]))
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        mark = "right" if row["right"] else ("NO ANSWER: " + row["error"][:80] if row["error"] else "wrong")
        say(f"  [{n}/{len(todo)}] {ex['id']}: {mark} ({row['seconds']:.1f}s)")
    return load_results(task, model)


def run(task: dict, models: list[str], url: str = ollama.DEFAULT_URL, ask=None, again: bool = False,
        limit: int = 0, say=print, check_installed: bool = True) -> None:  # fmt: skip
    """Every model in turn. Each is taken out of memory afterwards, so the next is measured on its own."""
    if check_installed:
        have = ollama.installed(url)
        missing = [m for m in models if m not in have and f"{m}:latest" not in have]
        if missing:
            raise ollama.OllamaError("Not downloaded yet: " + ", ".join(missing) + ". Run: "
                                     + "; ".join(f"ollama pull {m}" for m in missing))  # fmt: skip
    else:
        have = {}
    for model in models:
        say(f"{model}: {len(task['examples'][: limit or None])} examples")
        rows = run_model(task, model, ask=ask, url=url, again=again, limit=limit, say=say)
        info = have.get(model) or have.get(f"{model}:latest") or {}
        mem = ollama.memory(model, url) if check_installed else {}
        meta = {"model": model, "fingerprint": task["fingerprint"], "when": dt.datetime.now().isoformat(timespec="seconds"),
                "download_bytes": info.get("size", 0), "parameter_size": info.get("parameter_size", ""),
                "quantization": info.get("quantization", ""), "memory_bytes": mem.get("size", 0),
                "gpu_bytes": mem.get("size_vram", 0), "ollama": ollama.version(url) if check_installed else ""}  # fmt: skip
        old = load_meta(task, model)
        if old.get("fingerprint") == task["fingerprint"] and not meta["memory_bytes"]:
            meta["memory_bytes"], meta["gpu_bytes"] = old.get("memory_bytes", 0), old.get("gpu_bytes", 0)
        (results_dir(task) / f"{safe(model)}.meta.json").write_text(json.dumps(meta, indent=1))
        right = sum(r["right"] for r in rows)
        say(f"{model}: {right} of {len(rows)} right")
        if check_installed:
            ollama.unload(model, url)
