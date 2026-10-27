"""The answer to "which model is good enough?", from the saved results.

For each model: how many examples it got right, the range its real accuracy probably lies in (the
example count matters: 9 of 10 says much less than 90 of 100), how long one answer takes, and how much
memory it needs. Then a recommendation: the fastest model whose accuracy reaches your target, and
whether the best two really differ or could just be lucky on these examples.
"""

import csv
import json
from pathlib import Path

from . import runner, stats


def summary(task: dict, models: list[str] | None = None) -> list[dict]:
    """One row per model with results: right, n, accuracy, low, high, seconds, memory, verdict."""
    target = float(task["target"])
    out = []
    for model in models or runner.models_with_results(task):
        rows = runner.load_results(task, model)
        if not rows:
            continue
        k, n = sum(r["right"] for r in rows), len(rows)
        low, high = stats.wilson(k, n)
        meta = runner.load_meta(task, model)
        if k / n >= target and low >= target:
            verdict = "good enough"
        elif k / n >= target:
            verdict = "probably: test more examples"
        else:
            verdict = "not good enough"
        fields = {}
        for name in task.get("fields") or {}:
            fields[name] = sum(1 for r in rows if (r.get("fields") or {}).get(name)) / n
        out.append({"model": model, "right": k, "n": n, "accuracy": k / n, "low": low, "high": high,
                    "seconds": stats.median([r["seconds"] for r in rows]),
                    "tokens_per_second": stats.median([r["tokens_per_second"] for r in rows if r["tokens_per_second"]]),
                    "load_seconds": max((r.get("load_seconds", 0) for r in rows), default=0),
                    "no_answer": sum(1 for r in rows if r.get("error")),
                    "memory_gb": round((meta.get("memory_bytes") or meta.get("download_bytes") or 0) / 1e9, 1),
                    "gpu_share": (meta.get("gpu_bytes", 0) / meta["memory_bytes"]) if meta.get("memory_bytes") else 0,
                    "fields": fields, "verdict": verdict})  # fmt: skip
    return out


def compare(task: dict, a: str, b: str) -> dict:
    """Two models on the examples both answered: where only one was right, and McNemar's p-value."""
    ra = {r["id"]: r["right"] for r in runner.load_results(task, a)}
    rb = {r["id"]: r["right"] for r in runner.load_results(task, b)}
    both = set(ra) & set(rb)
    only_a = sum(1 for i in both if ra[i] and not rb[i])
    only_b = sum(1 for i in both if rb[i] and not ra[i])
    return {"a": a, "b": b, "only_a": only_a, "only_b": only_b, "shared": len(both), "p": stats.mcnemar(only_a, only_b)}


def recommend(task: dict, rows: list[dict]) -> dict:
    """{pick, why, compared}: the fastest model that is good enough, or the most accurate if none is."""
    if not rows:
        return {"pick": None, "why": "No results yet. Run: model-check run <task> -m <model>", "compared": None}
    target = float(task["target"])
    good = [r for r in rows if r["verdict"] == "good enough"]
    maybe = [r for r in rows if r["verdict"].startswith("probably")]
    best = sorted(rows, key=lambda r: (-r["accuracy"], r["seconds"]))
    compared = compare(task, best[0]["model"], best[1]["model"]) if len(best) > 1 else None
    if good:
        pick = min(good, key=lambda r: r["seconds"])
        why = (f"{pick['model']} reaches your target of {target:.0%} even at the low end of its range "
               f"({pick['low']:.0%}), and it is the fastest model that does ({pick['seconds']:.1f}s an answer).")
    elif maybe:
        pick = min(maybe, key=lambda r: r["seconds"])
        why = (f"{pick['model']} got {pick['accuracy']:.0%} right, above your target of {target:.0%}, but with "
               f"{pick['n']} examples its real accuracy could be as low as {pick['low']:.0%}. Add examples to be sure.")
    else:
        pick = best[0]
        why = (f"No model reached your target of {target:.0%}. The best was {pick['model']} with "
               f"{pick['accuracy']:.0%}. Look at its mistakes (model-check show), then improve the prompt, "
               f"try a bigger model, or ask whether the target is realistic.")
    return {"pick": pick["model"], "why": why, "compared": compared}


def compared_text(c: dict | None) -> str:
    if not c or c["only_a"] + c["only_b"] == 0:
        return ""
    words = "could easily be chance" if c["p"] >= 0.05 else "is unlikely to be chance"
    return (f"{c['a']} vs {c['b']}: on the {c['shared']} examples both answered, only {c['a']} was right on "
            f"{c['only_a']} and only {c['b']} on {c['only_b']}. A split like that {words} (p = {c['p']:.2f}).")


def table(task: dict, rows: list[dict]) -> str:
    head = ["Model", "Right", "Accuracy (95% range)", "Seconds an answer", "Memory", "Verdict"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in sorted(rows, key=lambda r: (-r["accuracy"], r["seconds"])):
        mem = f"{r['memory_gb']:.1f} GB" + (" (GPU)" if r["gpu_share"] > 0.9 else "") if r["memory_gb"] else "?"
        lines.append(f"| {r['model']} | {r['right']}/{r['n']} | {r['accuracy']:.0%} ({r['low']:.0%} to {r['high']:.0%}) | "
                     f"{r['seconds']:.1f} | {mem} | {r['verdict']} |")  # fmt: skip
    return "\n".join(lines)


def text(task: dict) -> str:
    """The report in Markdown (also what the terminal shows)."""
    rows = summary(task)
    rec = recommend(task, rows)
    out = [f"# {task['name']}: which model is good enough?", "",
           f"{len(task['examples'])} examples, target {float(task['target']):.0%} right.", ""]  # fmt: skip
    if rows:
        out += [table(task, rows), ""]
        if task.get("fields"):
            out += ["Right per field:", ""]
            out += [f"- {r['model']}: " + ", ".join(f"{k} {v:.0%}" for k, v in r["fields"].items()) for r in rows]
            out.append("")
        if any(r["no_answer"] for r in rows):
            out += ["No usable answer (counted as wrong): "
                    + ", ".join(f"{r['model']} {r['no_answer']}" for r in rows if r["no_answer"]), ""]
    out += [f"**Recommendation:** {rec['why']}", ""]
    if (said := compared_text(rec["compared"])):
        out += [said, ""]
    out += ["Seconds an answer: the median, without loading the model (loading happens once, then it stays in memory).",
            "The range: where the real accuracy probably lies (95% Wilson interval); fewer examples, wider range."]  # fmt: skip
    return "\n".join(out) + "\n"


def write(task: dict) -> tuple[Path, Path]:
    """report.md and results.csv (every example, every model) in the results folder."""
    folder = runner.results_dir(task)
    folder.mkdir(exist_ok=True)
    md = folder / "report.md"
    md.write_text(text(task), encoding="utf-8")
    models = runner.models_with_results(task)
    by_model = {m: {r["id"]: r for r in runner.load_results(task, m)} for m in models}
    path = folder / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "expected"] + [c for m in models for c in (f"{m} answer", f"{m} right", f"{m} seconds")])
        for ex in task["examples"]:
            row = [ex["id"], json.dumps(ex["expected"], ensure_ascii=False)]
            for m in models:
                r = by_model[m].get(ex["id"])
                if r:
                    ans = r["answer"].get("answer", r["answer"]) if isinstance(r["answer"], dict) and task["kind"] != "fields" else r["answer"]
                    row += [r["error"] or json.dumps(ans, ensure_ascii=False), int(r["right"]), r["seconds"]]
                else:
                    row += ["", "", ""]
            w.writerow(row)
    return md, path


def mistakes(task: dict, model: str, show_all: bool = False) -> str:
    """Each example the model got wrong: the input (shortened), the right answer and the model's."""
    rows = {r["id"]: r for r in runner.load_results(task, model)}
    if not rows:
        return f"No results for {model}. Run: model-check run {task['folder']} -m {model}"
    out = []
    for ex in task["examples"]:
        r = rows.get(ex["id"])
        if not r or (r["right"] and not show_all):
            continue
        given = r["error"] or json.dumps(r["answer"] if task["kind"] == "fields" else (r["answer"] or {}).get("answer"),
                                         ensure_ascii=False)  # fmt: skip
        wrong = [k for k, v in (r.get("fields") or {}).items() if not v]
        text_in = " ".join(str(ex["input"]).split())
        out.append(f"{ex['id']}: {'RIGHT' if r['right'] else 'WRONG'}{' (' + ', '.join(wrong) + ')' if wrong else ''}\n"
                   f"  input:    {text_in[:160]}{'...' if len(text_in) > 160 else ''}\n"
                   f"  expected: {json.dumps(ex['expected'], ensure_ascii=False)}\n  got:      {given}")  # fmt: skip
    return "\n".join(out) or f"{model} got every example right."
