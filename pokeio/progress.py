"""Progress reward terms and milestones, built purely from the spec.

Every term is a potential ``phi`` over game memory; the per-step reward is
``weight * (phi_t - phi_{t-1})``. With ``monotone`` (the default) phi is a
running max, so the agent can't farm a counter by toggling it (puffer's
``max_event_rew``; Pleines' PC-deposit and Leech-Seed hacks).

``state_score`` is the weighted sum of *state-function* terms only (bitcount,
value, sum_softcap) without per-episode baselines. It is comparable across
environments and is what the swarm uses to find the frontier.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokeio.memory import Memory, UnresolvedField
from pokeio.spec import Milestone, ProgressTerm

STATE_KINDS = {"bitcount", "value", "sum_softcap"}


@dataclass
class _TermState:
    prev: float = 0.0
    best: float = float("-inf")
    base: float = 0.0
    acc: float = 0.0  # ratio_gain accumulator
    last_ratio: float | None = None
    last_count: int | None = None
    seen: set | None = None


class Progress:
    def __init__(self, terms: list[ProgressTerm], milestones: list[Milestone], mem: Memory):
        self.mem = mem
        self.terms = []
        self.disabled: list[str] = []
        for t in terms:
            fields = [t.field, t.count_field, t.denom]
            if all(f is None or mem.has(f) for f in fields):
                self.terms.append(t)
            else:
                self.disabled.append(t.name)
        self.milestones = [m for m in milestones if all(mem.has(c.field) for c in m.when)]
        self.disabled += [m.name for m in milestones if m not in self.milestones]
        self.state = {t.name: _TermState() for t in self.terms}
        self.reached: dict[str, int] = {}  # milestone -> step first reached this episode
        self.totals = {t.name: 0.0 for t in self.terms}

    # -- raw values -------------------------------------------------------
    def _count(self, t: ProgressTerm) -> int | None:
        return self.mem.get(t.count_field) if t.count_field else None

    def _absolute(self, t: ProgressTerm) -> float:
        m = self.mem
        if t.kind == "bitcount":
            return float(m.popcount(t.field, t.ignore_bits))
        if t.kind == "value":
            return float(m.get(t.field))
        if t.kind == "sum_softcap":
            arr = m.array(t.field)
            n = self._count(t)
            s = float(arr[: max(0, min(n, len(arr)))].sum()) if n is not None else float(arr.sum())
            return s if s <= t.knee or t.knee <= 0 else t.knee + (s - t.knee) * t.slope
        raise ValueError(t.kind)

    def _ratio(self, t: ProgressTerm) -> float:
        n = self._count(t)
        num = self.mem.array(t.field)
        den = self.mem.array(t.denom)
        k = len(num) if n is None else max(0, min(n, len(num)))
        d = float(den[:k].sum())
        return float(num[:k].sum()) / d if d > 0 else 0.0

    def _phi(self, t: ProgressTerm, st: _TermState) -> float:
        if t.kind in STATE_KINDS:
            return self._absolute(t) - st.base
        if t.kind == "ratio_gain":
            r, n = self._ratio(t), self._count(t)
            if st.last_ratio is not None and n == st.last_count and r > st.last_ratio:
                st.acc += r - st.last_ratio
            st.last_ratio, st.last_count = r, n
            return st.acc
        if t.kind == "distinct":
            st.seen.add(self.mem.get(t.field))
            return float(len(st.seen))
        raise ValueError(t.kind)

    # -- episode control ------------------------------------------------------
    def rebase(self, step: int = 0) -> None:
        """Start a new accounting episode at the current game state (after a reset
        or a swarm state load). No reward is paid for progress already present."""
        for t in self.terms:
            st = self.state[t.name] = _TermState(seen=set())
            try:
                st.base = self._absolute(t) if t.kind in STATE_KINDS else 0.0
                phi = self._phi(t, st)
            except UnresolvedField:
                phi = 0.0
            st.prev = st.best = phi
        self.reached = {m.name: step for m in self.milestones if self.mem.all(m.when)}
        self.totals = {t.name: 0.0 for t in self.terms}

    def step(self, step: int) -> tuple[float, dict[str, float], list[str]]:
        """Return (total progress reward, per-term rewards, newly reached milestones)."""
        total = 0.0
        parts: dict[str, float] = {}
        for t in self.terms:
            st = self.state[t.name]
            phi = self._phi(t, st)
            if t.monotone:
                phi = max(phi, st.best)
                st.best = phi
            r = t.weight * (phi - st.prev)
            st.prev = phi
            if r != 0.0:
                parts[t.name] = r
                self.totals[t.name] += r
                total += r
        new = []
        for m in self.milestones:
            if m.name not in self.reached and self.mem.all(m.when):
                self.reached[m.name] = step
                new.append(m.name)
        return total, parts, new

    # -- frontier scoring -----------------------------------------------------
    def state_score(self) -> float:
        return sum(
            t.weight * self._absolute(t) for t in self.terms if t.kind in STATE_KINDS and t.swarm
        )

    def milestone_score(self) -> float:
        return float(sum(1 for m in self.milestones if self.mem.all(m.when)))
