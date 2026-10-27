"""model-check on the command line.

  model-check init my-task [--example email-triage|order-details|blank]   a task folder to start from
  model-check models                                   the models you have downloaded
  model-check run my-task -m qwen3:4b -m gemma3:12b    ask every example, score every answer
  model-check report my-task                           the comparison again, from saved results
  model-check show my-task -m qwen3:4b [--all]         the examples a model got wrong
"""

import argparse
import shutil
import sys
from importlib import resources
from pathlib import Path

from . import __version__, ollama, report, runner
from . import task as tasks

BLANK_TASK = '''# What you ask, what kind of answer you expect, and how good is good enough.
name = "My task"
kind = "choice"          # choice, number, exact, contains or fields
labels = ["yes", "no"]   # for choice: the answers allowed
target = 0.9             # good enough: at least 90% right

system = "You answer questions about short texts."
prompt = """Does this text ask for a reply?

{input}"""
'''
BLANK_EXAMPLES = '''{"id": "1", "input": "Can you call me back tomorrow?", "expected": "yes"}
{"id": "2", "input": "Thanks, all sorted now.", "expected": "no"}
'''


def _models(values: list[str]) -> list[str]:
    out = []
    for v in values:
        out += [m.strip() for m in v.split(",") if m.strip()]
    return list(dict.fromkeys(out))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="model-check", description="Which local AI model is good enough for your task?")
    p.add_argument("--version", action="version", version=f"model-check {__version__}")
    p.add_argument("--url", default=ollama.DEFAULT_URL, help="Ollama's address (or OLLAMA_URL)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("init", help="make a task folder to start from")
    s.add_argument("folder")
    s.add_argument("--example", default="blank", choices=["blank", "email-triage", "order-details"])
    sub.add_parser("models", help="the models you have downloaded")
    s = sub.add_parser("run", help="ask every example to every model and score the answers")
    s.add_argument("folder")
    s.add_argument("-m", "--model", action="append", required=True, help="a model (repeat, or a,b,c)")
    s.add_argument("--limit", type=int, default=0, help="only the first N examples (a quick try)")
    s.add_argument("--again", action="store_true", help="ask again even examples already answered")
    s = sub.add_parser("report", help="the comparison, from saved results")
    s.add_argument("folder")
    s = sub.add_parser("show", help="the examples a model got wrong")
    s.add_argument("folder")
    s.add_argument("-m", "--model", required=True)
    s.add_argument("--all", action="store_true", help="the right ones too")
    a = p.parse_args(argv)
    try:
        if a.command == "init":
            return _init(Path(a.folder), a.example)
        if a.command == "models":
            have = ollama.installed(a.url)
            if not have:
                print("No models yet. Download one, e.g.: ollama pull qwen3:4b")
            for name, m in sorted(have.items(), key=lambda kv: kv[1]["size"]):
                print(f"{name:40} {m['size'] / 1e9:5.1f} GB  {m['parameter_size']:>7}  {m['quantization']}")
            return 0
        task = tasks.load(Path(a.folder))
        if a.command == "run":
            runner.run(task, _models(a.model), url=a.url, again=a.again, limit=a.limit)
        if a.command in ("run", "report"):
            print()
            print(report.text(task))
            md, csv_path = report.write(task)
            print(f"Saved: {md} and {csv_path}")
        if a.command == "show":
            print(report.mistakes(task, a.model, a.all))
        return 0
    except (tasks.TaskError, ollama.OllamaError) as e:
        print(f"model-check: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. The answers so far are saved; run the same command to carry on.", file=sys.stderr)
        return 130


def _init(folder: Path, example: str) -> int:
    if (folder / "task.toml").exists():
        print(f"{folder} already has a task.toml; nothing changed.", file=sys.stderr)
        return 1
    folder.mkdir(parents=True, exist_ok=True)
    if example == "blank":
        (folder / "task.toml").write_text(BLANK_TASK, encoding="utf-8")
        (folder / "examples.jsonl").write_text(BLANK_EXAMPLES, encoding="utf-8")
    else:
        source = resources.files("model_check") / "tasks" / example
        for name in ("task.toml", "examples.jsonl"):
            with resources.as_file(source / name) as f:
                shutil.copy(f, folder / name)
    n = len(tasks.load(folder)["examples"])
    print(f"Made {folder}/task.toml and {folder}/examples.jsonl ({n} examples).")
    print(f"Next: model-check run {folder} -m <model> -m <another model>   (see your models: model-check models)")
    return 0
