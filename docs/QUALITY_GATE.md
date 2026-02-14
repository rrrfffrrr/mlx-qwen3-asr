# Quality Gate

This repo follows a strict order:

1. Match official quality/correctness.
2. Then optimize for Apple Silicon runtime.
3. Then port stable behavior to Swift.

## Modes

### Fast Gate (required for PRs)

```bash
python scripts/quality_gate.py --mode fast
```

Checks:
- Ruff lint on tracked Python files.
- Full pytest suite.

### Release Gate (required before tags/releases)

```bash
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release
```

Checks:
- Everything in fast gate.
- Reference parity test (`tests/test_reference_parity.py`).

### Nightly Regression Lane (scheduled/manual)

`Nightly Regression` workflow runs on macOS and tracks:
- Fast gate
- Golden quality sample (`scripts/eval_librispeech.py`)
- Latency/RTF benchmark (`scripts/benchmark_asr.py`)

This lane is intentionally separate from PR CI so day-to-day development stays fast.

## Pass Criteria

- `fast` gate must pass on every pull request.
- `release` gate must pass before publishing releases/artifacts.
- Nightly regression should remain green; red runs block performance claims until investigated.
- Any optimization PR that changes decoding/model math must include:
  - parity evidence (release gate pass or equivalent),
  - benchmark before/after results.

## Why This Exists

- Prevents silent quality regressions while optimizing.
- Keeps claims honest: parity first, speed second.
- Creates a clear handoff path for later Swift porting.
