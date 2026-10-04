# pokeIO

A game-agnostic reinforcement-learning agent for console games: it learns from the
screen, plus whatever RAM signals a per-game spec declares, and it does it in public
on a live training wall. Pokémon Yellow is the first challenge, not the product.

v3 is a rewrite built on what actually beat Pokémon Red (pokemonred_puffer,
Pleines et al. 2025). It drops v1's neuroevolution and v2's foveal-only perception.
The design rationale is in [docs/puffer-lessons.md](docs/puffer-lessons.md) and
[docs/sota-review-2026-10.md](docs/sota-review-2026-10.md).

> **Status:** this code has not been run yet. The Phase 0 bring-up checklist is in [TODO.md](TODO.md).

## How it works

- **Game spec** (`games/<game>/spec.yaml`): buttons, RAM fields, position, progress terms, milestones. It is the only game-specific file; the engine never names a game.
- **Env**: press 8 of 24 frames per action and auto-advance through text. The observation is a 2-bit packed screen plus a visited-cell memory channel.
- **Rewards**: a new-cell exploration reward, effective interactions, and spec-defined progress potentials. Exploration memory is wiped every ~20k steps while the game keeps running.
- **Learner**: recurrent PPO (CNN → LSTM, about 5M parameters) over an async shared-memory pool of hundreds of PyBoy instances.
- **Swarm**: when one agent reaches a new frontier, the population jumps to its save state.
- **Eval**: held-out milestone rates from the start state, with confidence intervals and a blind-screen check.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## The training wall

Every training run serves a live dashboard at `http://<host>:8600/`:

- one focus agent (what it sees, what it remembers, its action probabilities, value estimate and saliency)
- 16 agents playing at once
- the milestone ladder, with first-ever celebrations
- population footprints across every area discovered
- the swarm frontier gallery and an event feed
- learning vitals, reward anatomy, button usage, CPU and GPU load
- a counter of game time played, and a STOP button

To browse a finished run: `python -m pokeio dash --run runs/<name>`.

## Quick start

```bash
pip install -e ".[dev,tb]"         # Python ≥3.10; torch ≤2.14 for Pascal GPUs
pytest                             # ROM-free tests (gridworld fake console)
python -m pokeio train --spec games/gridworld/spec.yaml --config configs/smoke.yaml
```

Pokémon Yellow (supply your own ROM at `roms/pokemon_yellow.gb`):

```bash
python -m pokeio make-state --spec games/pokemon_yellow/spec.yaml   # boot -> new game state
python -m pokeio probe      --spec games/pokemon_yellow/spec.yaml   # verify RAM fields + overlay PNGs
python -m pokeio bench      --spec games/pokemon_yellow/spec.yaml --workers 52 --envs 6 --batch-workers 26
python -m pokeio train      --spec games/pokemon_yellow/spec.yaml --config configs/default.yaml --config configs/p100.yaml
python -m pokeio eval       --spec games/pokemon_yellow/spec.yaml --ckpt runs/<run>/ckpt/latest.pt --episodes 32 --steps 20000
python -m pokeio record     --spec games/pokemon_yellow/spec.yaml --ckpt runs/<run>/ckpt/latest.pt --out rec/
```

Override any config value with `--set section.key=value` (e.g. `--set train.device=cuda:1 --set dash.port=8601` for a second seed on the second GPU).

## Layout

```
pokeio/            engine (game-agnostic)
  spec.py          spec schema + loader          env.py        GameEnv
  memory.py        typed RAM fields              exploration.py cell memory, visited mask, novelty
  progress.py      reward terms + milestones     vec.py        async shared-memory env pool
  policy.py        CNN+LSTM actor-critic         train.py      recurrent PPO
  swarm.py         frontier state sharing        evaluate.py   held-out eval
  tools.py         probe / bench / make-state / record
  platforms/       gameboy (PyBoy), gridworld (fake)
  dash/            live hub, HTTP server, wall.html
games/             one spec per game (see games/README.md)
configs/           run configs (default, p100, smoke)
docs/              architecture, puffer lessons, SOTA review, archive of v1/v2
```

v1 (NEAT) and v2 (foveal PPO) are in git history and in `docs/archive/`.
