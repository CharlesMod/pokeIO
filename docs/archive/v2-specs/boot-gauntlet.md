# Spec: Boot Gauntlet — from-boot competence measurement (+ gated robustification)

**Status:** ready to implement (D1). D2/D3 are gated follow-ups — do NOT build them yet.
**Owner context:** written 2026-07-16 after the drop-in-generality discussion; companion TODO
entries live in `TODO.md` § "Boot Gauntlet".

---

## Motivation (why this exists)

Go-Explore restores mean a frontier cell at depth 40k was reached by a *chain* of episodes run by
possibly different genomes. **No single genome has ever demonstrated newgame→frontier play**, and
nothing currently measures whether the generation champion can still play from power-on:

- ~50% of players spawn from `reset_state` each wave (`restore_prob=0.5`,
  `_sample_restores`, `pokeio/train/loop.py:520`), and cohort rank normalization
  (`cohort_rank_normalize`, `pokeio/train/loop.py:816`) keeps selection pressure on newgame
  performance — so *forgetting the opening is selected against* — but we never observe how DEEP a
  single genome gets from boot.
- `replay_champion` (`pokeio/train/loop.py:897`) is a 48-step telemetry snippet, not an evaluation.

The boot gauntlet turns "we don't know" into a per-run curve: **how deep into the discovered
frontier can the current champion re-reach from `yellow_newgame.state`, solo, in one episode?**
This is the original Go-Explore paper's "robustification" concern, instrumented before we pay for
any robustification mechanism.

---

## D1 — the gauntlet (implement now)

### Behavior

Every `boot_gauntlet_every` generations (at the generation boundary, right after
`replay_champion` at `pokeio/train/loop.py:1773`), run the generation champion **solo** from
`reset_state` for `boot_gauntlet_steps` agent-steps at full speed (no realtime pacing), and score
the trajectory against the Go-Explore archive. Attach the result to that generation's
`GenerationRecord`.

### New module: `pokeio/train/gauntlet.py`

Keep it out of `loop.py` (that file is huge). Two functions:

```python
def boot_progress_from_keys(
    keys: Iterable[bytes],
    goexplore: GoExplore,
) -> tuple[int, int, int]:
    """Pure metric: (max_known_depth, n_distinct, n_known).

    max_known_depth = max(goexplore.cells[k].depth for visited k in cells), 0 if none.
    n_distinct      = count of distinct keys visited.
    n_known         = count of distinct visited keys present in goexplore.cells.
    """

def run_boot_gauntlet(
    genome, env, encoder, archive, goexplore, device,
    steps, max_nodes, max_conns, reset_state,
) -> dict[str, float]:
    """Run the champion solo from reset_state; return the metric dict (below)."""
```

`run_boot_gauntlet` mirrors the `replay_champion` loop (`pokeio/train/loop.py:897`): compile a
1-genome `Population`, `env.reset(reset_state)`, then per step encode → `population_forward_sparse`
(with `FORWARD_STEPS`) → argmax action → `env.step`. Collect
`archive.cell_key(screen, env-provided wram)` per step into a set, then call
`boot_progress_from_keys`.

**Key compatibility (verified, rely on it):** `NoveltyArchive.cell_key` on the parent-side env's
raw WRAM is byte-identical to the worker compact path (`cell_key_compact`,
`pokeio/reward/archive.py:142` — docstring guarantees it), and `GoExplore.cells` is keyed by
exactly these keys (`wave.last_key` → `goexplore.note`, `pokeio/train/loop.py:334`). No new
hashing code — use `archive.cell_key(screen, wram)` as-is.

### Metric dict (returned + logged)

| field | meaning |
|---|---|
| `boot_depth` | max `CellEntry.depth` among visited cells present in the Go-Explore archive |
| `boot_depth_frac` | `boot_depth / max(1, goexplore_max_depth)` — **the stall signal** |
| `boot_cells` | distinct cell keys visited during the gauntlet |
| `boot_cells_known` | distinct visited keys present in `goexplore.cells` |
| `boot_steps` | steps actually run (for interpretation when the knob changes mid-run) |

`goexplore_max_depth` comes from the same `go.stats()` snapshot taken for `reward_terms`
(`pokeio/train/loop.py:1848`) — same-snapshot consistency matters because eviction can shrink max
depth between generations.

**Metric caveat (document in the docstring):** `boot_depth` measures re-reaching *currently
archived* cells. A champion that finds a genuinely novel route from boot scores low on
`boot_cells_known` while being excellent — that's why `boot_cells` (raw distinct cells) is logged
alongside. Don't "fix" this by writing to the archive (see invariants).

### Invariants (non-negotiable)

1. **STRICTLY READ-ONLY** against both archives — same rule as `replay_champion` (audit REWARD#5,
   comment at `pokeio/train/loop.py:921`): no `archive.add/observe/visit`, no `goexplore.note`, no
   state captures. Membership probes on `goexplore.cells` and key construction only.
2. **Game-agnostic** — no RAM addresses, no game constants. Everything flows through existing
   digest machinery.
3. **Off the hot path** — runs at the generation boundary only, never inside a wave. At the
   Phase-0 single-core rate (~354 steps/s) a 4,096-step gauntlet is ~12s; at `every=10` and ~100s
   generations that's ~1% wall overhead. The fleet idles during it — acceptable; a subprocess
   variant is future work, don't build it now.
4. **No `time.sleep` pacing** — full speed. Keep the `time.sleep(0)` GIL-handoff trick from
   `replay_champion` if the live streamer is on.

### Wiring in `loop.py`

- `train(...)` gains `boot_gauntlet_every: int = 10` and `boot_gauntlet_steps: int = 0`
  (0 = auto → `4 * episode_steps`). `boot_gauntlet_every=0` disables.
- CLI: `--boot-gauntlet-every` (default 10), `--boot-gauntlet-steps` (default 0/auto), added next
  to `--restore-prob` (`pokeio/train/loop.py:1984`).
- Call site: generation boundary, after `replay_champion(...)` (`pokeio/train/loop.py:1773`),
  reusing `replay_env` (it's reset at the start of every replay/gauntlet, so sharing is safe).
  Guard: only when `go is not None` and `gen % boot_gauntlet_every == 0` (and gen > 0 if you want
  to skip the noise gen, implementer's choice — document it).
- If the streamer is on, `streamer.set_phase("evolving", "boot gauntlet")` around it.

### Telemetry

- Extend `GenerationRecord` (`pokeio/telemetry/schema.py:39`) with **defaulted** fields so old
  JSONL parses and old readers ignore new keys:
  `boot_depth: float = -1.0`, `boot_depth_frac: float = -1.0`, `boot_cells: float = -1.0`,
  `boot_cells_known: float = -1.0`, `boot_steps: float = 0.0`. `-1.0` = "gauntlet didn't run this
  gen" (distinguish from a genuine 0 score). Bump the schema version marker if one exists.
- **Also mirror the dict into `reward_terms`** (only on gens where it ran) — `reward_terms`
  already flows to the dashboard side panel via `streamer.set_side_stats`
  (`pokeio/train/loop.py:1862`) and into the generation JSONL, so the metric is visible on the
  wall with **zero dashboard work**. A dedicated wall chart (boot_depth_frac vs
  goexplore_max_depth over gens, in `pokeio/dash/wall.html`) is a stretch item, not required.
- Print one line at the boundary:
  `[gen N] boot-gauntlet: depth 1234/40000 (3.1%) cells 456 (known 321) in 4096 steps`.

### Tests (`tests/`)

- **Pure metric:** build a tiny synthetic `GoExplore` with a few `CellEntry`s at known depths;
  feed `boot_progress_from_keys` key sets covering: empty, no overlap, partial overlap, full
  overlap. Assert exact tuples.
- **Read-only guarantee:** run `run_boot_gauntlet` against a real 1-env setup (same pattern as
  existing integration smokes) and assert `archive.size`, `len(goexplore.cells)`,
  `goexplore.n_captured` are unchanged after the call.
- **Key-path equality** (cheap regression guard): one assertion that
  `archive.cell_key(screen, wram) == archive.cell_key_compact(screen, wram[::archive.wram_stride])`
  on a live env frame — this is the invariant the whole metric rests on.

### Acceptance criteria

1. A run with `--boot-gauntlet-every 5` produces generation records where every 5th record has
   `boot_depth >= 0` and the rest have `-1.0`; JSONL from an old run still loads via
   `read_generations`.
2. Dashboard side panel shows `boot_depth_frac` on gauntlet gens without any dashboard code change.
3. Archives provably untouched (test above).
4. Overhead: generation wall-time on gauntlet gens grows by roughly `boot_gauntlet_steps / 354`
   seconds, nothing else regresses (spot-check steps/s print).

---

## D2 — backward-shift robustification (GATED — do not implement yet)

**Gate:** only build after D1 data shows a stall — operationally, `boot_depth_frac` flat or
falling for ≥5 consecutive gauntlets while `goexplore_max_depth` grows ≥20% over the same window.

Design sketch (for when the gate trips): bias `_sample_restores` (`pokeio/train/loop.py:520`)
toward *shallower* cells for a fraction of restored players — e.g. sample those players' cells
from the bottom-q depth quantile with q annealed down as boot_depth_frac recovers. This is the
evolutionary analog of Go-Explore's "backward algorithm": force lineages to re-earn the chain from
progressively earlier spawns instead of only pushing the frontier. Knobs should live next to
`restore_prob`; the depth machinery already exists (`CellEntry.depth`,
`pokeio/reward/goexplore.py:68`; depth-weighted sampling in `_weights_vec`). No other design work
committed — revisit against real D1 curves.

---

## D3 — multi-game fitness (Phase 5, spec-only pointer)

For literal cross-game drop-in of a *single genome*: evaluate each genome on ≥2 ROMs and aggregate
per-game cohort ranks (extend `cohort_rank_normalize` to a (game × spawn-cohort) grid). Blocked on
the Phase-5 generalization acid test (Super Mario Land manifest regen) working at all. Recorded
here so the TODO checkbox has a home; no design commitment.
