"""
Populate the hidden ground-truth map with hazards.

Start and goal cells are reserved (kept safe) so a mission is always
at least attemptable. Everything else is fair game.
"""

import random
from config import HAZARD_COUNTS, START, GOAL


def populate_true_terrain(env, seed=None):
    if seed is not None:
        random.seed(seed)

    size = env.size
    reserved = {START, GOAL}

    def place(kind, n):
        placed = 0
        while placed < n:
            x, y = random.randint(0, size - 1), random.randint(0, size - 1)
            if (x, y) in reserved or env.get_type(x, y) != "safe":
                continue
            env.set_type(x, y, kind)
            placed += 1

    # craters first (most dangerous), then lava, then gas
    place("crater", HAZARD_COUNTS["crater"])
    place("lava", HAZARD_COUNTS["lava"])
    place("gas", HAZARD_COUNTS["gas"])
