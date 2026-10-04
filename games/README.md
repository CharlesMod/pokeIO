# Game specs

A spec is a YAML file: everything the engine needs to know about one game, and
the only place game knowledge is allowed. The fully commented reference is
`pokemon_yellow/spec.yaml`. The schema (with defaults) is in `pokeio/spec.py`.

| Spec | Purpose |
|---|---|
| `pokemon_yellow/` | The current challenge. |
| `pokemon_red/` | Cross-check against pokemonred_puffer and Pleines et al.: same engine, the game they report on. |
| `generic_gb/` | Pixels only, zero RAM knowledge: the generality floor for any Game Boy ROM. |
| `gridworld/` | A ROM-free fake console for tests and smoke runs. |

## Writing a spec for a new game

1. **Start from `generic_gb/spec.yaml`.** Set `rom`, the buttons, and `frames_per_action`: 24 suits turn-based RPGs; action games want 4 to 12.
2. **Build a start state.** Either record one by hand (any PyBoy save state works), or add a `start.boot_macro` and run `python -m pokeio make-state --spec ...`.
3. **Add a position.** Find the RAM bytes for room, x and y (a disassembly's `.sym` file, a RAM map wiki, or a memory search), then set `position.cell_px` and `camera`. This unlocks the visited mask and cell exploration, by far the most valuable signal.
4. **Add progress terms.** Story flags (`bitcount`), counters (`value`), stats (`sum_softcap`). Keep it to a handful; puffer's sweeps mostly tuned reward weights, not PPO.
5. **Add milestones**, which drive evaluation and the dashboard ladder, and choose a `swarm.score`.
6. **Run `python -m pokeio probe --spec ...`.** It prints every field at the start state, checks that the direction buttons move the player, and writes `probe/*.png`. Check that the visited-mask channel in `start_obs.png` lines up under the player sprite, and adjust `player_cell_px` if it doesn't.
7. **Run `python -m pokeio bench --spec ...`** for throughput, then train.

## Provenance rules

Mark every address in the YAML with where it came from: verified on the ROM,
derived, or symbol-only. Wrong addresses are silent poison. v1 trained for days
on off-by-one Pokémon Yellow addresses that read neighbouring bytes. Fields
declared with `symbol:` that don't resolve are disabled with a warning rather
than guessed. `probe` lists them.

## Yellow notes

- The Yellow WRAM layout in the 0xD000s is Red's minus 1. Every address verified on the ROM matches that rule. `[D]` addresses in the spec rely on it.
- `wJoyIgnore` (auto-advance through text) is symbol-only until a `pokeyellow.sym` is present. To build one: `git clone https://github.com/pret/pokeyellow && cd pokeyellow && make`, then copy `pokeyellow.sym` to `roms/`. Without it, the agent has to press A through text itself, which is slower but works.
- Pikachu can't learn Cut. Plan for that before expecting the agent past Vermilion: either a Cut-capable catch or the HM01 detour.
