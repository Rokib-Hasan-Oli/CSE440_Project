"""
Ground-truth volcanic terrain.

This holds the REAL map. The exploration agent never reads it directly --
it can only sense cells near its current position (see agent/explorer.py).
That separation is what makes the problem an exploration-under-uncertainty
task rather than a fully-observed shortest-path problem.
"""

import numpy as np
from config import GRID_SIZE, GOAL


class GridWorld:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        # terrain types: "safe", "lava", "gas", "crater", "goal"
        self.terrain = np.full((size, size), "safe", dtype=object)
        self.terrain[GOAL] = "goal"

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    # kept for API parity with the original project
    def is_valid(self, x, y):
        return self.in_bounds(x, y)

    def get_type(self, x, y):
        return self.terrain[x, y]

    def set_type(self, x, y, t):
        self.terrain[x, y] = t
