"""A task: what you ask the model, what kind of answer you expect, and your examples with the right answers.

A task is a folder with two files:

  task.toml        the question (with {input} where each example goes), the kind of answer, and how
                   good is good enough (target)
  examples.jsonl   one example per line: {"id": "...", "input": "...", "expected": ...}
                   (optionally "images": ["photo.jpg"], paths inside the folder, for vision models)

Kinds of answer, each checked by plain code, never by another model:

  choice    one label from a list (sorting e-mails, tickets, reviews)
  number    a number, right within a tolerance (an amount, a count)
  exact     a short text that must match (a name, a code), ignoring case and spacing
  contains  a text that must mention every expected word or phrase
  fields    several named values at once (extracting details), each with its own way to match
"""

import hashlib
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

KINDS = ("choice", "number", "exact", "contains", "fields")
MATCHES = ("exact", "number", "similar", "date", "choice")


class TaskError(ValueError):
    """The task folder has a mistake: the message says where."""


def load(folder: Path) -> dict:
    """The task, checked, with its examples: {name, kind, prompt, ..., examples: [...], folder, fingerprint}."""
    folder = Path(folder)
    if not (folder / "task.toml").exists():
        raise TaskError(f"No task.toml in {folder}. Make a starting point with: model-check init {folder}")
    try:
        task = tomllib.loads((folder / "task.toml").read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise TaskError(f"task.toml: {e}") from None
    task["folder"] = folder
    kind = task.get("kind")
    if kind not in KINDS:
        raise TaskError(f"task.toml: kind must be one of {', '.join(KINDS)} (it is {kind!r})")
    if "{input}" not in task.get("prompt", ""):
        raise TaskError("task.toml: the prompt needs {input} where each example's text goes")
    task.setdefault("name", folder.name)
    task.setdefault("system", "")
    task.setdefault("target", 0.9)
    if not 0 < float(task["target"]) <= 1:
        raise TaskError("task.toml: target is a share between 0 and 1, e.g. 0.9 for 90% right")
    if kind == "choice":
        labels = task.get("labels") or []
        if len(labels) < 2:
            raise TaskError("task.toml: a choice task needs labels = [...] with at least two")
        task["labels"] = [str(x) for x in labels]
    if kind == "number":
        task.setdefault("tolerance", 0.005)  # within half a cent, for money
    if kind == "fields":
        fields = task.get("fields") or {}
        if not fields:
            raise TaskError("task.toml: a fields task needs a [fields] table, e.g. quantity = { type = \"number\" }")
        for name, f in fields.items():
            f.setdefault("type", "string")
            if f["type"] not in ("string", "number", "date"):
                raise TaskError(f"task.toml: field {name}: type must be string, number or date")
            f.setdefault("match", {"number": "number", "date": "date"}.get(f["type"], "exact"))
            if f["match"] not in MATCHES:
                raise TaskError(f"task.toml: field {name}: match must be one of {', '.join(MATCHES)}")
            if f["match"] == "choice" and len(f.get("labels") or []) < 2:
                raise TaskError(f"task.toml: field {name}: match = \"choice\" needs labels = [...]")
    task["examples"] = examples(folder, task)
    task["fingerprint"] = fingerprint(folder)
    return task


def examples(folder: Path, task: dict) -> list[dict]:
    path = folder / "examples.jsonl"
    if not path.exists():
        raise TaskError(f"No examples.jsonl in {folder}: one example per line, with its right answer")
    out, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
        except json.JSONDecodeError as e:
            raise TaskError(f"examples.jsonl line {n}: not valid JSON ({e.msg})") from None
        if "input" not in ex or "expected" not in ex:
            raise TaskError(f"examples.jsonl line {n}: needs \"input\" and \"expected\"")
        ex["id"] = str(ex.get("id") or n)
        if ex["id"] in seen:
            raise TaskError(f"examples.jsonl line {n}: the id {ex['id']} is used twice")
        seen.add(ex["id"])
        for image in ex.get("images") or []:
            if not (folder / image).exists():
                raise TaskError(f"examples.jsonl line {n}: no image {image} in {folder}")
        if task["kind"] == "choice" and str(ex["expected"]) not in task["labels"]:
            raise TaskError(f"examples.jsonl line {n}: expected {ex['expected']!r} isn't one of the labels")
        if task["kind"] == "fields":
            missing = [f for f in task["fields"] if f not in (ex["expected"] or {})]
            if missing:
                raise TaskError(f"examples.jsonl line {n}: expected has no {', '.join(missing)}")
        out.append(ex)
    if not out:
        raise TaskError("examples.jsonl has no examples")
    return out


def fingerprint(folder: Path) -> str:
    """Changes when task.toml or examples.jsonl change: results from before then are out of date."""
    h = hashlib.sha256()
    for name in ("task.toml", "examples.jsonl"):
        h.update((folder / name).read_bytes())
    return h.hexdigest()[:12]


def schema(task: dict) -> dict:
    """The JSON schema the answer must follow (Ollama's structured outputs)."""
    kind = task["kind"]
    if kind == "fields":
        props = {}
        for name, f in task["fields"].items():
            p = {"type": "number" if f["type"] == "number" else "string"}
            if f.get("labels"):
                p["enum"] = [str(x) for x in f["labels"]]
            if f.get("description"):
                p["description"] = f["description"]
            props[name] = p
        return {"type": "object", "properties": props, "required": list(props)}
    answer = {"type": "number"} if kind == "number" else {"type": "string"}
    if kind == "choice":
        answer["enum"] = task["labels"]
    return {"type": "object", "properties": {"answer": answer}, "required": ["answer"]}


def prompt(task: dict, example: dict) -> str:
    """The question for one example, with a last line saying the shape of the answer (it helps the
    model, and Ollama's documentation suggests it)."""
    text = task["prompt"].replace("{input}", str(example["input"]))  # replace, not format: inputs have braces
    kind = task["kind"]
    if kind == "choice":
        shape = f'Answer in JSON as {{"answer": one of {", ".join(task["labels"])}}}.'
    elif kind == "number":
        shape = 'Answer in JSON as {"answer": a number, with no units or thousands separators}.'
    elif kind == "fields":
        parts = []
        for name, f in task["fields"].items():
            what = "YYYY-MM-DD" if f["type"] == "date" else "a number" if f["type"] == "number" else "text"
            if f.get("labels"):
                what = "one of " + ", ".join(str(x) for x in f["labels"])
            parts.append(f'"{name}": {what}')
        shape = "Answer in JSON as {" + ", ".join(parts) + "}."
    else:
        shape = 'Answer in JSON as {"answer": "..."}.'
    return f"{text}\n\n{shape}"
