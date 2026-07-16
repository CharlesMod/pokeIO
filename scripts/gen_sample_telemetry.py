"""Emit synthetic GenerationRecords to runs/_sample/ for testing the LIVE wall.

    python scripts/gen_sample_telemetry.py [n]

Uses the real telemetry schema/writer so the shape matches a genuine run.
"""

import math
import os
import sys

from pokeio.telemetry.schema import GenerationRecord, TelemetryWriter

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
RUN = "runs/_sample"
os.makedirs(RUN, exist_ok=True)
# start clean
open(os.path.join(RUN, "telemetry.jsonl"), "w").close()

w = TelemetryWriter(RUN)
best = 200.0
cells = 0
for g in range(1, N + 1):
    best += max(0.0, 55 * math.sin(g * 0.3) + (40 if g % 7 == 0 else 6) + g * 2.5)
    med = best * 0.62
    worst = best * 0.22
    delta = (8 if g % 5 else 44) + (g % 9)
    cells += delta
    w.write_generation(
        GenerationRecord(
            gen=g,
            wall_time=g * 4.2,
            fitness_best=round(best, 1),
            fitness_median=round(med, 1),
            fitness_worst=round(worst, 1),
            n_species=8 + (g % 6),
            archive_cells=cells,
            archive_delta=delta,
            champion_id="#" + str(4400 + (g * 7 % 600)),
            champion_genome_ref="g" + str(g),
            reward_terms={"novelty": round(best * 0.4, 1)},
            throughput_sps=22000 + (g * 137) % 4000,
            cpu_pct=90.0,
            gpu=[
                {"index": 0, "util": 10.0, "mem_used_mb": 13750, "mem_total_mb": 16384},
                {"index": 1, "util": 60.0, "mem_used_mb": 4100, "mem_total_mb": 16384},
            ],
        )
    )
w.close()
print(f"wrote {N} generations to {RUN}/telemetry.jsonl")
