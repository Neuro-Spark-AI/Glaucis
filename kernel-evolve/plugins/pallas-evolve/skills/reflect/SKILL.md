---
name: reflect
description: Use when recording batch kernel optimization learnings to AGENT.md — iterates over N variant results, extracts failure patterns and successful optimizations, posts round summary to GitHub Issue
---

# Record Batch Optimization Learnings

After each batch evolution round, iterate over all variant results, extract learnings, and record them to AGENT.md. Post a round summary to the GitHub Issue.

## Context

Invoked by `pallas-evolve:start` after `pallas-evolve:analyze`, or standalone. Expects the batch round directory `iteration_{N}/variants/*/eval_result.json`, along with `iteration_{N}/batch_analysis.md` and `iteration_{N}/selection.md`.

## Procedure

### Step 1: Read iteration data

Read:
- `iteration_{N}/variants/*/eval_result.json` — evaluation results for ALL variants in this round
- `iteration_{N}/batch_analysis.md` — comparative analysis across variants
- `iteration_{N}/selection.md` — lineage selection decisions (promotions, prunings)
- `iteration_{N}/strategy.md` — the optimization directions attempted this round
- the learnings file at `{session.agent_md_path}` (default `AGENT.md` at the repo root) — existing learnings

### Step 2: Determine learnings across variants

Iterate over each variant's result. Not every variant produces a learning. Only record when:

**Failure worth recording** (new pattern not already in AGENT.md):
- A compilation error with a non-obvious cause
- A correctness error that reveals a Pallas/TPU constraint
- A timeout or infrastructure failure with a reproducible cause

**Success worth recording** (meaningful improvement):
- Speedup improved >=5% over the previous best
- A specific technique produced a measurable improvement
- The improvement reveals a generalizable optimization principle

**Comparative learnings** (patterns across variants or rounds):
- Direction X consistently outperforms direction Y across rounds
- A mutation type reliably produces improvements vs. another
- Lineage characteristics that predict success

Skip recording if:
- The failure is a trivial syntax error (typo, missing import)
- The speedup change is negligible (<5% improvement)
- A similar pattern already exists in AGENT.md (check by reading existing entries)

**Deduplication within the round**: Multiple variants may hit the same failure or discover the same optimization. Only record each unique pattern once, noting which variants exhibited it.

### Step 3: Update AGENT.md (if new learnings)

The learnings file lives at `{session.agent_md_path}` (default `AGENT.md` at the
repo root). For private kernels this should point into a private-repo checkout so
learnings — which quote kernel code and results — never land in a public engine
fork. If it doesn't exist, create it with:

```markdown
# Pallas Kernel Optimization Agent Knowledge

## Failure Patterns

## Successful Optimizations
```

**For failures**, find the next available `[Fxxx]` number and append under `## Failure Patterns`:

```markdown
### [F{NNN}] {Short description}
- **Symptom**: {What the error looks like — include key error message text}
- **Root cause**: {Why it happens — the underlying Pallas/TPU constraint}
- **Fix**: {How to avoid it in future kernels}
- **First seen**: {YYYY-MM-DD}, {kernel_name} optimization
```

**For successful optimizations**, find the next `[Sxxx]` number and append under `## Successful Optimizations`:

```markdown
### [S{NNN}] {Short description of the technique}
- **Optimization**: {What was changed in the kernel code}
- **Impact**: {before_speedup}x -> {after_speedup}x on {shape}
- **Why it works**: {Root cause analysis — why this specific change improved performance}
- **Applicable when**: {Conditions where this technique should be tried again}
- **First seen**: {YYYY-MM-DD}, {kernel_name} optimization
```

**Deduplication**: Before adding, read all existing entries. If a similar pattern exists:
- For failures: update the existing entry with additional context rather than creating a duplicate
- For successes: update with the new performance data if the technique is the same

Batch all updates into a single AGENT.md write — do not make multiple separate edits.

### Step 4: Comment on GitHub Issue

Post a batch round summary comment to the tracking Issue. If
`session.tracking_repo` is set, pass `--repo {session.tracking_repo}` so the
comment (which contains variant code/results) lands in the private repo, not a
public fork:

```bash
gh issue comment {issue_number} ${TRACKING_REPO:+--repo "$TRACKING_REPO"} --body "$(cat <<'EOF'
### Round {N} Summary

| Variant | Status | Speedup | Direction | Notable |
|---------|--------|---------|-----------|---------|
| {variant_name} | {SUCCESS/COMPILE_ERROR/INCORRECT} | {speedup}x | {direction} | {brief note} |
| ... | ... | ... | ... | ... |

**Active Lineages:** {lineage_id} ({speedup}x, {direction}), ...
**Pruned this round:** {pruned_variant_names or "none"}
**New learnings:** {[F/S NNN] entries or "none"}
EOF
)"
```

### Step 5: Commit AGENT.md changes (if any)

If the learnings file was modified, commit it **in the repo that owns it** — the
one containing `{session.agent_md_path}`. When that path is inside a private-repo
checkout, commit there; NEVER commit learnings to a public engine fork.

```bash
AGENT_MD="{session.agent_md_path}"
git -C "$(dirname "$AGENT_MD")" add "$(basename "$AGENT_MD")"
git -C "$(dirname "$AGENT_MD")" commit -m "docs(agent): record learnings from {kernel_name} round {N}"
```

(If `agent_md_path` is not inside a git checkout you control, skip the commit —
the file still persists on disk for the next run.)

### Step 6: Context note

All learnings have been persisted:
- `AGENT.md` — updated failure patterns and successful optimizations (committed)
- GitHub Issue #{issue_number} — round summary comment posted

**Do NOT compact here** — this skill is typically invoked within the `pallas-evolve:start` loop (Phase 4), and the subsequent Phase 5 (COMPACT) handles context compression with proper state verification. Compacting mid-loop would lose orchestration context.

If invoked **standalone** (outside the start loop), invoke `/compact` after this skill completes.
