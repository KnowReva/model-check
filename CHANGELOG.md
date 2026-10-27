# Changelog

## 0.1.0 (2026-10-27)

- `init`, `models`, `run`, `report`, `show`.
- Kinds of answer: choice, number, exact, contains, fields (exact, similar, number, date, choice).
- Structured outputs for every answer; plain-code scoring; several right answers allowed.
- Accuracy with its 95% Wilson range, a verdict against your target, the fastest good-enough model,
  and McNemar's exact test between the two most accurate.
- Results saved per example as they come (a stopped run carries on); a changed task sets old
  results aside; each model out of memory after its turn.
- Two practice tasks: email-triage (24 e-mails) and order-details (20 order messages).
