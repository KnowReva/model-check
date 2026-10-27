"""Is an answer right? Plain code, the same way every time, so two models are judged alike."""

import datetime as dt
import difflib
import re


def norm(text) -> str:
    """Lower case, single spaces, no punctuation at the ends: "  Acme Ltd. " and "acme ltd" match."""
    return re.sub(r"\s+", " ", str(text if text is not None else "")).strip().strip(".,;:!?\"'").casefold()


def number(value) -> float | None:
    """A number from a number or a text like "1,250.00" or "$45"; None when there is none."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d[\d,]*\.?\d*", str(value or ""))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def date(value) -> str:
    """YYYY-MM-DD from a few common ways of writing a date; the text itself when it isn't one."""
    text = str(value or "").strip()
    for f in ("%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y", "%d %b %Y", "%b %d, %Y"):
        try:
            return dt.datetime.strptime(text, f).date().isoformat()
        except ValueError:
            continue
    return text


def same(got, expected, match: str, tolerance: float = 0.005) -> bool:
    """One value against the right one, by a way of matching. A list of right answers: any of them."""
    if isinstance(expected, list):
        return any(same(got, e, match, tolerance) for e in expected)
    if match == "number":
        a, b = number(got), number(expected)
        return a is not None and b is not None and abs(a - b) <= tolerance
    if match == "date":
        return date(got) == date(expected)
    if match == "similar":  # names: small spelling differences are fine
        return difflib.SequenceMatcher(None, norm(got), norm(expected)).ratio() >= 0.85
    return norm(got) == norm(expected)  # exact, choice


def score(task: dict, answer: dict | None, expected) -> dict:
    """{right: bool, fields: {name: bool}} for one answer (answer None: the model gave none)."""
    kind = task["kind"]
    if answer is None:
        return {"right": False, "fields": {f: False for f in task.get("fields") or {}}}
    if kind == "fields":
        fields = {name: same(answer.get(name), expected.get(name), f["match"], float(f.get("tolerance", 0.005)))
                  for name, f in task["fields"].items()}  # fmt: skip
        return {"right": all(fields.values()), "fields": fields}
    got = answer.get("answer")
    if kind == "number":
        right = same(got, expected, "number", float(task["tolerance"]))
    elif kind == "contains":  # a list here means every one of these words
        wanted = expected if isinstance(expected, list) else [expected]
        right = all(norm(w) in norm(got) for w in wanted)
    else:  # choice, exact (a list of right answers: any of them)
        right = same(got, expected, "exact")
    return {"right": right, "fields": {}}
