"""pokeIO standing credibility / eval harness (audit 2026-07-17).

Three research-motivated controls that must stand alongside any headline claim:

* **noise / blank-frame ablation** — is the champion actually *seeing*, or
  winning by action-timing? (:mod:`pokeio.eval.controls`)
* **random-weight-search baseline** — does structured search beat best-of-K
  random weights at all? (:mod:`pokeio.eval.controls`)
* **geometric-mean-of-milestones** — a Crafter-style headline with rliable-style
  IQM + bootstrap CIs, never a raw score. (:mod:`pokeio.eval.metrics`)

:mod:`pokeio.eval.harness` wires them to a run's checkpoint + telemetry and the
``scripts/eval_run.py`` CLI drives it. Fully isolated new code; dependency-light
(numpy + torch + repo modules); ROM-guarded.

:mod:`pokeio.eval.spine` (v2.0 #29) is the evaluation protocol on top of this
harness: from-boot competence as the headline (IQM+CIs), the same controls as
Goodhart catchers in the headline currency, a frozen-core git-diff guard, and a
pre-registered 2nd-game hook.
"""

from pokeio.eval import controls, harness, metrics, spine

__all__ = ["controls", "metrics", "harness", "spine"]
