import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from model_check import cli, ollama, report, runner, scoring, stats
from model_check import task as tasks


def make_task(folder: Path, kind="choice", examples=None, **extra) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f'kind = "{kind}"', 'prompt = """Sort this: {input}"""', "target = 0.8"]
    if kind == "choice":
        lines.append('labels = ["a", "b"]')
    for k, v in extra.items():
        lines.append(f"{k} = {json.dumps(v)}")
    (folder / "task.toml").write_text("\n".join(lines) + "\n")
    examples = examples or [{"id": str(i), "input": f"text {i}", "expected": "a" if i % 2 else "b"} for i in range(1, 11)]
    (folder / "examples.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
    return tasks.load(folder)


def fake_ask(wrong=(), seconds=1.0):
    """A model that answers from the example's text: right, except for the ids in `wrong`."""

    def ask(model, system, prompt, schema, images=()):
        i = int(prompt.split("text ")[1].split()[0])
        right = "a" if i % 2 else "b"
        answer = ({"a": "b", "b": "a"}[right]) if str(i) in wrong else right
        return {"answer": {"answer": answer}, "seconds": seconds, "load_seconds": 2.0 if i == 1 else 0.0,
                "tokens": 5, "tokens_per_second": 20.0}

    return ask


# ------------------------------------------------------------------ the task ---
def test_the_sample_tasks_load_and_their_answers_fit_their_schema():
    for name in ("email-triage", "order-details"):
        t = tasks.load(Path(tasks.__file__).parent / "tasks" / name)
        assert len(t["examples"]) >= 20 and t["fingerprint"]
        s = tasks.schema(t)
        assert s["type"] == "object" and set(s["required"]) == set(s["properties"])
    triage = tasks.load(Path(tasks.__file__).parent / "tasks" / "email-triage")
    assert tasks.schema(triage)["properties"]["answer"]["enum"] == ["billing", "support", "sales", "spam"]
    assert tasks.prompt(triage, {"input": "Refund {please}"}).endswith('Answer in JSON as {"answer": one of billing, support, sales, spam}.')
    assert "Refund {please}" in tasks.prompt(triage, {"input": "Refund {please}"})  # braces in an input are kept
    orders = tasks.load(Path(tasks.__file__).parent / "tasks" / "order-details")
    assert '"deliver_by": YYYY-MM-DD' in tasks.prompt(orders, {"input": "x"})
    assert tasks.schema(orders)["properties"]["quantity"] == {"type": "number"}


def test_mistakes_in_a_task_folder_are_said_plainly(tmp_path):
    with pytest.raises(tasks.TaskError, match="No task.toml"):
        tasks.load(tmp_path)
    make_task(tmp_path / "t")
    (tmp_path / "t" / "examples.jsonl").write_text('{"id": "1", "input": "x", "expected": "c"}\n')
    with pytest.raises(tasks.TaskError, match="line 1: expected 'c' isn't one of the labels"):
        tasks.load(tmp_path / "t")
    (tmp_path / "t" / "examples.jsonl").write_text('{"id": "1", "input": "x", "expected": "a"}\n{"id": "1", "input": "y", "expected": "b"}\n')
    with pytest.raises(tasks.TaskError, match="id 1 is used twice"):
        tasks.load(tmp_path / "t")
    (tmp_path / "t" / "task.toml").write_text('kind = "choice"\nlabels = ["a", "b"]\nprompt = "no placeholder"\n')
    with pytest.raises(tasks.TaskError, match=r"needs \{input\}"):
        tasks.load(tmp_path / "t")


# ------------------------------------------------------------------ scoring ---
def test_scoring_by_plain_code():
    assert scoring.same(" Acme Ltd. ", "acme ltd", "exact")
    assert scoring.same("$1,250.00", 1250, "number") and not scoring.same("1250.1", 1250, "number")
    assert scoring.same("14 March 2027", "2027-03-14", "date") and scoring.same("March 14, 2027", "2027-03-14", "date")
    assert scoring.same("Northside Cafe", "Northside Café", "similar") and not scoring.same("Northside", "Southside Bakery", "similar")
    assert scoring.same("bath towels", ["bath towels", "white bath towels"], "similar")  # any of the right answers
    t = {"kind": "fields", "fields": {"qty": {"match": "number"}, "who": {"match": "similar"}}}
    got = scoring.score(t, {"qty": "24", "who": "Tom"}, {"qty": 24, "who": "Northside Cafe"})
    assert got == {"right": False, "fields": {"qty": True, "who": False}}
    assert scoring.score({"kind": "contains"}, {"answer": "Call the bank and cancel the card"}, ["bank", "cancel"])["right"]
    assert scoring.score({"kind": "choice"}, None, "a") == {"right": False, "fields": {}}  # no answer: wrong


# ------------------------------------------------------------------ statistics ---
def test_how_sure_the_numbers_are():
    low, high = stats.wilson(18, 20)
    assert (round(low, 3), round(high, 3)) == (0.699, 0.972)  # 18 of 20: somewhere from 70% to 97%
    low, high = stats.wilson(180, 200)
    assert high - low < 0.09  # ten times the examples: a much narrower range
    assert stats.wilson(0, 0) == (0.0, 1.0)
    assert stats.mcnemar(3, 1) == pytest.approx(0.625)  # 3 against 1: could easily be chance
    assert stats.mcnemar(10, 0) < 0.01  # 10 against 0: hardly chance
    assert stats.mcnemar(0, 0) == 1.0 and stats.median([3, 1, 2, 10]) == 2.5


# ------------------------------------------------------------------ a run ---
def test_a_run_scores_saves_resumes_and_recommends(tmp_path):
    t = make_task(tmp_path / "t")
    said = []
    runner.run(t, ["small"], ask=fake_ask(wrong={"3", "4"}, seconds=0.5), say=said.append, check_installed=False, limit=4)
    assert len(runner.load_results(t, "small")) == 4
    runner.run(t, ["small", "big"], ask=fake_ask(wrong={"3", "4"}, seconds=0.5), say=said.append, check_installed=False)
    assert len(runner.load_results(t, "small")) == 10  # carried on from example 5, didn't ask 1-4 again
    assert said.count("small: 10 examples") == 1 and "  [1/6] 5: right (0.5s)" in said
    assert runner.models_with_results(t) == ["small", "big"]
    rows = {r["model"]: r for r in report.summary(t)}
    assert rows["small"]["right"] == 8 and rows["small"]["verdict"].startswith("probably")  # 80%, but could be 49%
    assert rows["small"]["load_seconds"] == 2.0 and rows["small"]["seconds"] == 0.5
    text = report.text(t)
    assert "| small | 8/10 | 80% (49% to 94%) | 0.5 |" in text
    assert "Add examples to be sure" in text
    md, csv_path = report.write(t)
    assert md.exists() and csv_path.read_text().splitlines()[0] == "id,expected,small answer,small right,small seconds,big answer,big right,big seconds"
    assert "3: WRONG" in report.mistakes(t, "small") and "1: RIGHT" not in report.mistakes(t, "small")


def test_the_fastest_good_enough_model_is_picked_and_a_real_difference_is_named(tmp_path):
    examples = [{"id": str(i), "input": f"text {i}", "expected": "a" if i % 2 else "b"} for i in range(1, 61)]
    t = make_task(tmp_path / "t", examples=examples)
    runner.run(t, ["slow-perfect"], ask=fake_ask(seconds=4.0), say=lambda m: None, check_installed=False)
    runner.run(t, ["fast-good"], ask=fake_ask(wrong={"2", "7"}, seconds=1.0), say=lambda m: None, check_installed=False)
    runner.run(t, ["tiny"], ask=fake_ask(wrong={str(i) for i in range(1, 25)}, seconds=0.2), say=lambda m: None,
               check_installed=False)  # fmt: skip
    rec = report.recommend(t, report.summary(t))
    assert rec["pick"] == "fast-good" and "fastest model that does (1.0s an answer)" in rec["why"]
    c = report.compare(t, "slow-perfect", "fast-good")
    assert (c["only_a"], c["only_b"]) == (2, 0) and "could easily be chance" in report.compared_text(c)
    c = report.compare(t, "slow-perfect", "tiny")
    assert "is unlikely to be chance" in report.compared_text(c)


def test_a_changed_task_sets_old_results_aside(tmp_path):
    t = make_task(tmp_path / "t")
    runner.run(t, ["m"], ask=fake_ask(), say=lambda m: None, check_installed=False)
    (tmp_path / "t" / "task.toml").write_text((tmp_path / "t" / "task.toml").read_text().replace("0.8", "0.9"))
    t2 = tasks.load(tmp_path / "t")
    assert t2["fingerprint"] != t["fingerprint"] and runner.load_results(t2, "m") == []
    said = []
    runner.run(t2, ["m"], ask=fake_ask(), say=said.append, check_installed=False, limit=1)
    assert any("the task changed" in s for s in said) and list((tmp_path / "t" / "results").glob("m.before-*.jsonl"))


def test_a_broken_answer_counts_as_wrong_and_the_run_goes_on(tmp_path):
    t = make_task(tmp_path / "t")

    def ask(**kw):
        raise ollama.BadAnswer("the answer wasn't valid JSON", {"seconds": 3.0})

    runner.run(t, ["m"], ask=ask, say=lambda m: None, check_installed=False, limit=2)
    rows = runner.load_results(t, "m")
    assert [r["right"] for r in rows] == [False, False] and rows[0]["seconds"] == 3.0
    assert "No usable answer (counted as wrong): m 2" in report.text(t)


# ------------------------------------------------------------------ Ollama itself ---
class FakeOllama(BaseHTTPRequestHandler):
    asked: list = []

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": "qwen3:4b", "size": 2_500_000_000,
                                         "details": {"parameter_size": "4.0B", "quantization_level": "Q4_K_M"}}]})  # fmt: skip
        elif self.path == "/api/ps":
            self._send(200, {"models": [{"name": "qwen3:4b", "size": 3_100_000_000, "size_vram": 0}]})
        elif self.path == "/api/version":
            self._send(200, {"version": "0.12.0"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.asked.append((self.path, body))
        if body.get("model") == "nothere":
            self._send(404, {"error": "model 'nothere' not found"})
        elif self.path == "/api/generate":
            self._send(200, {"done": True})
        else:
            self._send(200, {"message": {"content": '{"answer": "a"}'}, "done_reason": "stop",
                             "total_duration": 3_000_000_000, "load_duration": 1_000_000_000,
                             "eval_count": 10, "eval_duration": 500_000_000})  # fmt: skip


@pytest.fixture()
def server():
    s = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    FakeOllama.asked = []
    yield f"http://127.0.0.1:{s.server_port}"
    s.shutdown()


def test_talking_to_ollama(server, tmp_path):
    got = ollama.ask("qwen3:4b", "sys", "q", {"type": "object"}, url=server)
    assert got == {"answer": {"answer": "a"}, "seconds": 2.0, "load_seconds": 1.0, "tokens": 10, "tokens_per_second": 20.0}
    body = FakeOllama.asked[0][1]
    assert body["format"] == {"type": "object"} and body["options"]["temperature"] == 0 and body["think"] is False
    with pytest.raises(ollama.OllamaError, match="Run: ollama pull nothere"):
        ollama.ask("nothere", "", "q", {}, url=server)
    t = make_task(tmp_path / "t")
    with pytest.raises(ollama.OllamaError, match="Not downloaded yet: gemma3:12b. Run: ollama pull gemma3:12b"):
        runner.run(t, ["qwen3:4b", "gemma3:12b"], url=server, say=lambda m: None)
    runner.run(t, ["qwen3:4b"], url=server, say=lambda m: None, limit=2)
    meta = runner.load_meta(t, "qwen3:4b")
    assert meta["memory_bytes"] == 3_100_000_000 and meta["ollama"] == "0.12.0" and meta["quantization"] == "Q4_K_M"
    assert FakeOllama.asked[-1] == ("/api/generate", {"model": "qwen3:4b", "keep_alive": 0})  # out of memory after
    assert "3.1 GB" in report.text(t)


def test_the_command_line(tmp_path, capsys, server):
    assert cli.main(["init", str(tmp_path / "triage"), "--example", "email-triage"]) == 0
    assert "(24 examples)" in capsys.readouterr().out
    assert cli.main(["init", str(tmp_path / "triage")]) == 1  # never overwrites
    assert cli.main(["--url", server, "models"]) == 0
    assert "qwen3:4b" in capsys.readouterr().out
    assert cli.main(["--url", server, "run", str(tmp_path / "triage"), "-m", "qwen3:4b", "--limit", "3"]) == 0
    out = capsys.readouterr().out
    assert "| qwen3:4b | 0/3 | 0% (0% to 56%)" in out and "No model reached your target of 90%" in out
    assert cli.main(["--url", server, "show", str(tmp_path / "triage"), "-m", "qwen3:4b"]) == 0
    assert 'expected: "billing"' in capsys.readouterr().out
    assert cli.main(["report", str(tmp_path / "nothing")]) == 1
    assert cli.main(["init", str(tmp_path / "mine")]) == 0 and tasks.load(tmp_path / "mine")["kind"] == "choice"


def test_an_answer_wrapped_in_words_or_fences_still_counts():
    assert ollama.extract_json('```json\n{"answer": "a"}\n```') == {"answer": "a"}
    assert ollama.extract_json('Sure: {"answer": "b"} done') == {"answer": "b"}
    assert ollama.extract_json("<think>x</think> no json here") is None
