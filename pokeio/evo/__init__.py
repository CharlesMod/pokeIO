"""pokeIO evolution core — tensorized, GPU-batched NEAT (Phase 2).

The population's networks evaluate as tensors on card 1 (``forward.py``), while
topology-evolving operators (``ops.py``) mutate, breed, and speciate the genomes
(``genome.py``).  See ``demo_xor.py`` for the end-to-end XOR benchmark and the
throughput probe.

Structured for the HyperNEAT/ES-HyperNEAT future: ``forward.propagate`` and
``forward.propagate_sparse`` accept a ready weight tensor, so a CPPN that paints
a substrate can plug straight in.
"""

from pokeio.evo.forward import (
    CompiledPopulation,
    apply_activation,
    build_weight_matrix,
    population_forward,
    population_forward_sparse,
    propagate,
    propagate_sparse,
)
from pokeio.evo.genome import (
    ConnGene,
    Genome,
    InnovationTracker,
    NodeGene,
    Population,
    make_genome,
)
from pokeio.evo.ops import (
    MutationRates,
    Speciation,
    compatibility_distance,
    crossover,
    mutate_genome,
    reproduce,
    tournament_select,
)

__all__ = [
    # genome
    "Genome",
    "NodeGene",
    "ConnGene",
    "InnovationTracker",
    "Population",
    "make_genome",
    # forward
    "CompiledPopulation",
    "population_forward",
    "population_forward_sparse",
    "propagate",
    "propagate_sparse",
    "build_weight_matrix",
    "apply_activation",
    # ops
    "MutationRates",
    "Speciation",
    "mutate_genome",
    "crossover",
    "compatibility_distance",
    "tournament_select",
    "reproduce",
]
