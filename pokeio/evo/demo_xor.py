"""End-to-end NEAT sanity benchmark: evolve networks to solve XOR, plus a GPU
throughput probe at the real optical input scale.

Run::

    python -m pokeio.evo.demo_xor              # XOR + throughput probe on cuda:1
    python -m pokeio.evo.demo_xor --xor-only
    python -m pokeio.evo.demo_xor --probe-only

This exercises the whole pipeline: batched forward on card 1, mutation,
crossover, speciation, and selection.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch

from pokeio.evo.forward import (
    build_weight_matrix,
    population_forward,
    population_forward_sparse,
    propagate,
)
from pokeio.evo.genome import (
    HIDDEN,
    ConnGene,
    NodeGene,
    Population,
    InnovationTracker,
    make_genome,
)
from pokeio.evo.ops import MutationRates, Speciation, reproduce

XOR_INPUTS = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32
)
XOR_TARGETS = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)


def pick_device(prefer: str = "cuda:1") -> torch.device:
    if torch.cuda.is_available():
        idx = int(prefer.split(":")[1]) if ":" in prefer else 0
        if idx < torch.cuda.device_count():
            return torch.device(prefer)
        return torch.device("cuda:0")
    return torch.device("cpu")


@dataclass
class XORResult:
    solved: bool
    generations: int
    best_fitness: float
    best_error: float
    device: str
    n_hidden: int


def evaluate_xor(pop: Population, device: torch.device, steps: int = 12):
    """Return (fitness[N], error[N], correct[N]) for the population on XOR."""
    cp = pop.compile(device)
    X = XOR_INPUTS.to(device).unsqueeze(0).expand(pop.n, -1, -1)  # (N, 4, 2)
    out = population_forward(cp, X, steps=steps)  # (N, 4, 1)
    pred = out[:, :, 0]  # (N, 4)
    tgt = XOR_TARGETS.to(device).unsqueeze(0)  # (1, 4)
    err = (pred - tgt).abs()  # (N, 4)
    sum_err = err.sum(dim=1)  # (N,)
    fitness = (4.0 - sum_err) ** 2
    correct = ((pred > 0.5).float() == tgt).all(dim=1)  # (N,)
    return fitness.cpu(), sum_err.cpu(), correct.cpu()


def run_xor(
    seed: int = 0,
    pop_size: int = 150,
    max_gens: int = 300,
    device_str: str = "cuda:1",
    verbose: bool = True,
) -> XORResult:
    device = pick_device(device_str)
    rng = np.random.default_rng(seed)

    tracker = InnovationTracker(n_in=2, n_out=1)
    genomes = [
        make_genome(2, 1, tracker, rng, connect="full", weight_scale=1.0)
        for _ in range(pop_size)
    ]

    rates = MutationRates(
        add_node=0.03,
        add_conn=0.08,
        weight=0.9,
        weight_perturb_sigma=0.6,
        weight_reset_prob=0.1,
        weight_reset_scale=1.5,
        toggle=0.01,
        feedforward=True,
    )
    spec = Speciation(threshold=3.0, c1=1.0, c2=1.0, c3=0.5)

    best_fit = -1.0
    best_err = 4.0
    best_hidden = 0
    for gen in range(max_gens):
        pop = Population.from_genomes(genomes, max_nodes=64, max_conns=512)
        fitness, err, correct = evaluate_xor(pop, device)
        for i, g in enumerate(genomes):
            g.fitness = float(fitness[i])

        gi = int(torch.argmax(fitness))
        if float(fitness[gi]) > best_fit:
            best_fit = float(fitness[gi])
            best_err = float(err[gi])
            best_hidden = len(genomes[gi].hidden_ids())

        if bool(correct.any()):
            gi = int(torch.argmax(correct.float() * fitness))
            if verbose:
                print(
                    f"[XOR] SOLVED at gen {gen} "
                    f"(fitness={float(fitness[gi]):.3f}, err={float(err[gi]):.3f}, "
                    f"hidden={len(genomes[gi].hidden_ids())}, device={device})"
                )
            return XORResult(
                True, gen, float(fitness[gi]), float(err[gi]), str(device),
                len(genomes[gi].hidden_ids()),
            )

        if verbose and gen % 20 == 0:
            print(
                f"[XOR] gen {gen:3d}  best_fit={best_fit:.3f}  best_err={best_err:.3f}"
                f"  species={len(spec.reps)}  hidden(best)={best_hidden}"
            )

        species = spec.assign(genomes, rng)
        genomes = reproduce(
            genomes, species, tracker, rng, rates,
            pop_size=pop_size, tournament_size=3, elitism=1,
            crossover_rate=0.75, fitness_sharing=True,
            weight_scale=1.0, survival_threshold=0.4, c3=0.5,
        )

    if verbose:
        print(f"[XOR] NOT solved in {max_gens} gens (best_err={best_err:.3f})")
    return XORResult(False, max_gens, best_fit, best_err, str(device), best_hidden)


def _make_probe_genome(
    n_in: int, n_out: int, n_hidden: int, n_conn: int,
    tracker: InnovationTracker, rng: np.random.Generator,
):
    """A representative sparse evolved-ish genome: n_hidden hidden nodes and
    n_conn random feed-forward-ish connections over a huge input space."""
    g = make_genome(n_in, n_out, tracker, rng, connect="none")
    hidden_ids = []
    for _ in range(n_hidden):
        nid = tracker._next_node
        tracker._next_node += 1
        g.nodes[nid] = NodeGene(nid, HIDDEN, act=1, bias=0.0)
        hidden_ids.append(nid)
    inputs = list(range(n_in))
    outs = list(g.output_ids())
    for _ in range(n_conn):
        s = int(rng.choice(inputs)) if rng.random() < 0.7 or not hidden_ids else int(rng.choice(hidden_ids))
        d = int(rng.choice(hidden_ids + outs)) if hidden_ids else int(rng.choice(outs))
        if s == d:
            continue
        innov = tracker.conn_innov(s, d)
        g.conns[innov] = ConnGene(s, d, float(rng.normal(0, 1)), True, innov)
    return g


def throughput_probe(
    device_str: str = "cuda:1",
    pop_size: int = 256,
    n_in: int = 6144,
    n_out: int = 8,
    n_hidden: int = 64,
    n_conn: int = 512,
    steps: int = 4,
    chunk: int = 16,
    iters: int = 20,
) -> float:
    """Report net-evaluations/sec for pop_size genomes at the real optical dim.

    A "net evaluation" = one genome producing one action from one observation
    (a single forward through all ``steps`` of propagation).  A dense ``(M, M)``
    adjacency for the whole population at this input dim would be ~39 GB, so we
    process the population in genome-chunks: build each chunk's adjacency, run
    the propagation steps, and free it before the next chunk.  This is the
    honest steady-state cost including adjacency construction.
    """
    device = pick_device(device_str)
    rng = np.random.default_rng(0)
    tracker = InnovationTracker(n_in=n_in, n_out=n_out)
    genomes = [
        _make_probe_genome(n_in, n_out, n_hidden, n_conn, tracker, rng)
        for _ in range(pop_size)
    ]
    M = n_in + 1 + n_out + n_hidden
    pop = Population.from_genomes(genomes, max_nodes=M, max_conns=n_conn + 8)
    cp = pop.compile(device)

    X = torch.randn(pop_size, 1, n_in, device=device)  # one observation / genome

    print(
        f"[PROBE] pop={pop_size} n_in={n_in} n_out={n_out} hidden={n_hidden} "
        f"conns~{n_conn} steps={steps} device={device}"
    )

    def bench(fn, label: str) -> float:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        fn()  # warmup
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        eps = pop_size * iters / dt
        mem = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda"
            else 0.0
        )
        print(
            f"[PROBE] {label:6s}: {eps:>12,.0f} net-evals/sec  "
            f"({dt / iters * 1e3:6.1f} ms/pass, peak {mem:4.1f} GB)"
        )
        return eps

    # dense adjacency (chunked so the (M,M) tensors fit): the pathological path
    # at this input width — the input x input block is ~99% zeros.
    def dense_pass():
        for a in range(0, pop_size, chunk):
            b = min(a + chunk, pop_size)
            W = build_weight_matrix(cp, a, b)
            propagate(
                W, cp.node_act[a:b], cp.node_bias[a:b],
                cp.n_in, cp.n_out, X[a:b], steps,
            )
            del W

    # sparse edge-list (the right representation for wide, sparse optical nets).
    def sparse_pass():
        population_forward_sparse(cp, X, steps=steps)

    dense_eps = bench(dense_pass, "dense")
    sparse_eps = bench(sparse_pass, "sparse")
    print(
        f"[PROBE] sparse is {sparse_eps / dense_eps:.0f}x the dense path and "
        f"{sparse_eps / 10000:.0f}x the ~10k emu-steps/s CPU ceiling."
    )
    return sparse_eps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xor-only", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=150)
    ap.add_argument("--max-gens", type=int, default=300)
    args = ap.parse_args()

    if not args.probe_only:
        run_xor(
            seed=args.seed, pop_size=args.pop, max_gens=args.max_gens,
            device_str=args.device,
        )
    if not args.xor_only:
        throughput_probe(device_str=args.device)


if __name__ == "__main__":
    main()
