from __future__ import annotations

import itertools
from typing import Dict, Iterable, Iterator, List

import numpy as np


def parameter_grid(param_space: Dict[str, Iterable]) -> List[Dict]:
    keys = list(param_space.keys())
    combos = itertools.product(*(param_space[k] for k in keys))
    return [dict(zip(keys, values)) for values in combos]


def parameter_random(
    param_space: Dict[str, Iterable],
    n_samples: int,
    seed: int = 42,
) -> Iterator[Dict]:
    rng = np.random.default_rng(seed)
    keys = list(param_space.keys())
    values = {k: list(v) for k, v in param_space.items()}

    for _ in range(n_samples):
        yield {k: values[k][rng.integers(0, len(values[k]))] for k in keys}
