"""Periodic training checkpoints so a crash / power loss doesn't discard a run.

A power event once cost a 94k-deep Go-Explore frontier that lived only in the
trainer's memory. This snapshots the full evolutionary state to disk every N
generations and can resume from it, turning a blip from "start over at gen 0"
into "lose < N generations".

What's saved (everything needed to continue bit-for-bit from the boundary):
  * the population (next generation's genomes)
  * the Go-Explore archive — cells + their emulator save_state blobs (the heavy
    payload; a full 16k-cell archive is ~3 GB, so the checkpoint is large)
  * the novelty archive (seen keys + visit counts + per-gen frontier)
  * the innovation tracker, speciation state, species lineage bookkeeping
  * the main RNG (numpy Generator preserves its bit-generator state)
  * mined progress-counter taps + the miner's rollout deque + exclude set
  * the resolved gen to resume at

The write is atomic (tmp + os.replace), so a crash mid-write leaves the
previous good checkpoint intact. It runs SYNCHRONOUSLY at the generation
boundary where these objects are quiescent (no wave is mutating them), which
costs a short stall proportional to the archive size — the price of not losing
overnight work. Keep the interval coarse enough (config.run.checkpoint_every_gens)
that the stall is a small fraction of wall time.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

CHECKPOINT_FILENAME = "checkpoint.pkl"
_SCHEMA = 1

# Load-status codes returned by :func:`load_checkpoint_status`. A ``--resume``
# caller must treat ABSENT (no checkpoint yet — fine to start fresh) very
# differently from CORRUPT / SCHEMA (a checkpoint is present but unusable —
# restarting at gen 0 would silently discard the run and clobber its telemetry).
LOAD_OK = "ok"
LOAD_ABSENT = "absent"
LOAD_CORRUPT = "corrupt"
LOAD_SCHEMA = "schema_mismatch"


def checkpoint_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / CHECKPOINT_FILENAME


def save_checkpoint(run_dir: str | Path, state: dict) -> float:
    """Atomically write ``state`` to ``<run_dir>/checkpoint.pkl``.

    Returns the seconds the write took (for the boundary profiler). ``state``
    is a plain dict of picklable objects (see :func:`build_state`).
    """
    t0 = time.perf_counter()
    path = checkpoint_path(run_dir)
    tmp = path.with_suffix(".pkl.tmp")
    payload = {"schema": _SCHEMA, **state}
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())  # durability: survive a power cut right after
    os.replace(tmp, path)  # atomic: readers see old-or-new, never partial
    return time.perf_counter() - t0


def load_checkpoint_status(run_dir: str | Path) -> tuple[dict | None, str]:
    """Load a checkpoint AND report *why* nothing loaded.

    Returns ``(data, status)`` where ``status`` is one of :data:`LOAD_OK`,
    :data:`LOAD_ABSENT`, :data:`LOAD_CORRUPT`, or :data:`LOAD_SCHEMA`. This lets
    a ``--resume`` path distinguish "no checkpoint yet" (safe to start fresh)
    from "checkpoint present but unreadable" (must abort, never silently
    restart at gen 0). The technical reason is printed here; callers add the
    user-facing decision.
    """
    path = checkpoint_path(run_dir)
    if not path.exists():
        return None, LOAD_ABSENT
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    except Exception as e:  # corrupt / truncated (a crash mid-fsync is possible)
        print(f"[checkpoint] load failed ({e})", flush=True)
        return None, LOAD_CORRUPT
    if data.get("schema") != _SCHEMA:
        print(f"[checkpoint] schema {data.get('schema')} != {_SCHEMA}", flush=True)
        return None, LOAD_SCHEMA
    return data, LOAD_OK


def load_checkpoint(run_dir: str | Path) -> dict | None:
    """Load a checkpoint, or ``None`` if absent / unreadable / wrong schema.

    Backward-compatible wrapper over :func:`load_checkpoint_status` (which a
    ``--resume`` path should prefer so it can tell absent from corrupt).
    """
    data, status = load_checkpoint_status(run_dir)
    if status in (LOAD_CORRUPT, LOAD_SCHEMA):
        print("[checkpoint] ignoring unusable checkpoint; starting fresh", flush=True)
    return data


def build_state(
    *,
    gen: int,
    genomes,
    archive,
    goexplore,
    tracker,
    spec,
    prev_reps,
    species_best,
    rng,
    taps,
    miner_rollouts,
    miner_exclude,
    total_agent_steps=0,
) -> dict:
    """Assemble the checkpoint dict from the loop's live objects.

    ``gen`` is the NEXT generation to run on resume (the loop checkpoints after
    reproduction, so ``genomes`` is already the next population).
    """
    return {
        "gen": int(gen),
        "genomes": genomes,
        "archive": archive,
        "goexplore": goexplore,
        "tracker": tracker,
        "spec": spec,
        "prev_reps": prev_reps,
        "species_best": species_best,
        "rng_state": rng.bit_generator.state,
        "taps": taps,
        "miner_rollouts": list(miner_rollouts),
        "miner_exclude": miner_exclude,
        "total_agent_steps": int(total_agent_steps),
    }


__all__ = [
    "CHECKPOINT_FILENAME",
    "checkpoint_path",
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_status",
    "LOAD_OK",
    "LOAD_ABSENT",
    "LOAD_CORRUPT",
    "LOAD_SCHEMA",
    "build_state",
]
