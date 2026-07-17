"""Terminal sparklines + trend helpers for the analytics report.

Pure formatting — no game or ML knowledge. Kept apart from report.py so the
tiny bits (sparklines, deltas, humanized durations) are unit-testable.
"""

from __future__ import annotations

from typing import Sequence

_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], width: int = 48) -> str:
    """Unicode-block sparkline; downsamples to ``width`` buckets (mean per bucket)."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return ""
    if len(vals) > width:
        # bucket-average down to width points
        n, out = len(vals), []
        for i in range(width):
            lo = i * n // width
            hi = max(lo + 1, (i + 1) * n // width)
            chunk = vals[lo:hi]
            out.append(sum(chunk) / len(chunk))
        vals = out
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return _BLOCKS[0] * len(vals)
    return "".join(
        _BLOCKS[min(len(_BLOCKS) - 1, int((v - lo) / span * (len(_BLOCKS) - 1)))]
        for v in vals
    )


def humanize_secs(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


def fmt_int(n: float) -> str:
    return f"{int(n):,}"


def trend_arrow(values: Sequence[float], window: int = 10) -> str:
    """↑/↓/→ over the last ``window`` samples (by first-vs-last mean split)."""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return "→"
    tail = vals[-window:]
    half = max(1, len(tail) // 2)
    a = sum(tail[:half]) / half
    b = sum(tail[-half:]) / half
    if b > a * 1.05:
        return "↑"
    if b < a * 0.95:
        return "↓"
    return "→"


__all__ = ["sparkline", "humanize_secs", "fmt_int", "trend_arrow"]
