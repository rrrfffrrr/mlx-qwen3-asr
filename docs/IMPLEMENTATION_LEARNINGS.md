# Implementation Learnings

Purpose: capture concise, high-signal implementation memory for future coding
agents.

## Rules

1. Log non-trivial implementation slices only.
2. Prefer reusable heuristics over timeline narration.
3. Keep entries concise and evidence-backed.
4. If no material failures occurred, state that explicitly.

## Retrospective Card

### Core Fields (required)

1. `Scope`: what changed and where.
2. `Decision`: what was kept and why (with tradeoff).
3. `What worked`: highest-signal patterns/commands (max 3 bullets).
4. `What did not work`: failed paths or false assumptions with root cause.
5. `Reuse next time`: command/lane/path/pattern to copy directly.
6. `Evidence`: tests/artifacts/benchmarks/commits proving the result.

### Optional Fields

1. `Risk left`: unresolved risk or quality gap.
2. `Revisit trigger`: concrete condition/date to reopen this decision.
3. `ROI`: `high`, `medium`, or `low`.

## Entry Template

```md
### YYYY-MM-DD - <Change Slice>
- Scope: <what changed and where>
- Decision:
  - <what we kept, why, and primary tradeoff>
- What worked:
  - <high-signal command/pattern/result>
- What did not work:
  - <failed path/assumption + root cause>
- Reuse next time:
  - <command/lane/path/pattern>
- Evidence:
  - <tests/artifacts/benchmark IDs/commit IDs>
- Risk left: <optional>
- Revisit trigger: <optional>
- ROI: <optional high|medium|low>
```

## Entries

### 2026-02-16 - Standardize learnings format for future agents
- Scope: Added Retrospective Card policy in `CLAUDE.md` and created this
  canonical log file.
- Decision:
  - Keep a single implementation learnings file with required core fields;
    tradeoff is small process overhead in exchange for better cross-agent
    continuity and faster onboarding.
- What worked:
  - Defining required vs optional fields preserved flexibility without losing
    consistency.
  - Making `Reuse next time` mandatory forced extraction of actionable insight.
- What did not work:
  - Prior guidance was too open-ended, so notes risked becoming generic and
    hard to reuse.
- Reuse next time:
  - When changing process policy, update both `CLAUDE.md` and this log in the
    same commit.
- Evidence:
  - Commit touching `CLAUDE.md` and `docs/IMPLEMENTATION_LEARNINGS.md`
- ROI: high
