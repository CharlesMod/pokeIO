from pathlib import Path

import pytest

from pokeio.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def grid_spec():
    return load_spec(ROOT / "games/gridworld/spec.yaml", ROOT)


@pytest.fixture
def root():
    return ROOT
