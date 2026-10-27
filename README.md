# model-check

Which local AI model is good enough for **your** task? Write down a few dozen examples with their
right answers, and model-check asks every model you have, scores every answer with plain code, and
tells you the fastest model that is right often enough, with an honest range for how sure that is.
Everything runs on your own computer with [Ollama](https://ollama.com).

```
$ model-check run email-triage -m qwen3:4b -m gemma3:12b -m llama3.2:3b
...
| Model       | Right | Accuracy (95% range) | Seconds an answer | Memory | Verdict                      |
|---|---|---|---|---|---|
| gemma3:12b  | 23/24 | 96% (80% to 99%)     | 6.2               | 9.1 GB | probably: test more examples |
| qwen3:4b    | 22/24 | 92% (74% to 98%)     | 1.9               | 3.4 GB | probably: test more examples |
| llama3.2:3b | 17/24 | 71% (51% to 85%)     | 1.4               | 2.8 GB | not good enough              |
```

(An illustration of the layout, not a measurement: run it on your own computer for your numbers.)

A [KnowReva](https://knowreva.com) open-source project. The guide
[Choose a local AI model by measuring it](https://knowreva.com/guides/) explains every part of the
code step by step.

## Why measure

Leaderboards rank models on other people's questions. Your task is yours: sorting your customers'
e-mails, pulling the details out of your orders, reading your invoices. A small model is often good
enough for one job and useless for another, and the only way to know is to try it on your own
examples. model-check makes that a habit: the same examples, the same scoring, every model.

## Install

You need [Ollama](https://ollama.com) with at least one model, and Python 3.10 or newer.

```bash
ollama pull qwen3:4b                     # a small model to start with (2.5 GB)
pip install git+https://github.com/knowreva/model-check
```

## Use

```bash
model-check init email-triage --example email-triage   # a ready practice task: sorting 24 e-mails
model-check models                                      # the models you have, smallest first
model-check run email-triage -m qwen3:4b -m gemma3:12b  # every example through every model
model-check show email-triage -m qwen3:4b               # the ones it got wrong, next to the right answer
model-check report email-triage                         # the comparison again, from the saved results
```

`--limit 5` tries the first five examples only. A run that is stopped carries on where it was; `--again`
asks everything again. Results are saved in the task's `results/` folder: one file per model, a
`report.md` and a `results.csv` with every answer from every model side by side.

Settings: `--url` (or `OLLAMA_URL`, default `http://localhost:11434`), `MODEL_CHECK_CONTEXT` (the
context window in tokens, for long inputs; not sent unless set).

## Your own task

A task is a folder with two files. `task.toml` says what you ask and what kind of answer you expect:

```toml
name = "Sort customer e-mails"
kind = "choice"
labels = ["billing", "support", "sales", "spam"]
target = 0.9   # good enough: at least 90% right

system = "You sort the e-mails a small online shop receives."
prompt = """Which team should handle this e-mail?

E-mail:
{input}"""
```

`examples.jsonl` holds one example per line, with its right answer:

```json
{"id": "e01", "input": "I was charged twice for order 4471.", "expected": "billing"}
{"id": "e02", "input": "The lid of my kettle won't stay closed.", "expected": "support"}
```

| `kind` | The answer | Right when |
|---|---|---|
| `choice` | one of `labels` | it is the expected label |
| `number` | a number | within `tolerance` (default 0.005) |
| `exact` | a short text | the same, ignoring case, spaces and end punctuation |
| `contains` | a text | it mentions every expected word or phrase (`"expected": ["refund", "5 days"]`) |
| `fields` | several values | every field matches, each its own way (see below) |

For `fields`, list them in `task.toml`; `type` is `string`, `number` or `date` (answered as
YYYY-MM-DD), and `match` is `exact`, `similar` (names: small spelling differences are fine), `number`,
`date` or `choice` (with `labels`):

```toml
[fields]
customer = { type = "string", match = "similar" }
quantity = { type = "number" }
deliver_by = { type = "date" }
```

An expected value can be a list of right answers when more than one is fair:
`"product": ["bath towels", "white bath towels"]`. Vision models: add `"images": ["photo.jpg"]` to an
example (paths inside the task folder).

The answer always comes back in a fixed JSON shape (Ollama's structured outputs), so scoring never
depends on how chatty a model is. A model that doesn't answer with valid JSON gets that example
wrong, and the report counts those separately.

## Reading the report

- **Right** and **Accuracy**: the share of examples with the right answer, and the range the real
  accuracy probably lies in (the 95% Wilson interval). With 24 examples, 22 right is "somewhere from
  74% to 98%": the range, not the single number, is what the examples can tell you.
- **Seconds an answer**: the median, without the one-off loading of the model.
- **Memory**: what the model took in memory while answering (from Ollama); `(GPU)` when it ran on a
  graphics card.
- **Verdict**: `good enough` when even the low end of the range reaches your target; `probably` when
  the accuracy reaches it but the range dips below (add examples); `not good enough` otherwise.
- **Recommendation**: the fastest model that is good enough. It also says whether the two most
  accurate models really differ, using only the examples where exactly one of them was right
  (McNemar's exact test): 3 against 1 "could easily be chance"; 10 against 0 "is unlikely to be".

Each model is taken out of memory after its turn (`keep_alive` 0), so the next one is measured on its
own, and answers are asked with `temperature` 0 and a fixed seed, so a run can be repeated.

## Two practice tasks

- `email-triage`: 24 made-up e-mails to a small online shop, sorted into billing, support, sales
  and spam, including a few that could fool a model (a phishing "mailbox full" message, a refund
  question that sounds like support).
- `order-details`: 20 made-up order messages; the customer, the product, the quantity (some in
  words: "two dozen") and the delivery date (written many ways) are pulled out.

They are written to show how model-check works; your own examples are the ones that answer your
question.

## Tests

```bash
pip install -e '.[test]'
pytest
```

The tests use a fake Ollama, so they run anywhere in a second or two.

## Licence

MIT
