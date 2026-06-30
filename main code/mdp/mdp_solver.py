"""
Stochastic MDP solver (Value Iteration).

This is the key upgrade over a deterministic grid solver: transitions are
PROBABILISTIC. An action succeeds with probability (1 - slip); otherwise the
agent slips to one of the two perpendicular cells. The Bellman backup
therefore averages over outcomes, which makes the optimal policy naturally
keep a safety margin away from craters and lava.

    V(s)  = max_a  sum_s'  P(s'|s,a) * [ R(s') + gamma * V(s') ]
    pi(s) = argmax_a  ( same )

Goal and known craters are absorbing (terminal) states with V = 0; their
reward is collected by whichever neighbour transitions into them.
"""

import numpy as np
from config import ACTIONS, ACTION_DELTAS, GAMMA, VI_ITERATIONS, VI_TOLERANCE

# perpendicular ("slip") directions for each intended action
PERP = {
    "UP": ["LEFT", "RIGHT"],
    "DOWN": ["LEFT", "RIGHT"],
    "LEFT": ["UP", "DOWN"],
    "RIGHT": ["UP", "DOWN"],
}


class MDPSolver:
    def __init__(self, size):
        self.size = size

    def _move(self, x, y, action):
        dx, dy = ACTION_DELTAS[action]
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.size and 0 <= ny < self.size:
            return nx, ny
        return x, y  # bumping a wall keeps you in place

    def _outcomes(self, x, y, action, slip):
        """Stochastic transition distribution: list of (prob, nx, ny)."""
        outs = {}
        ix, iy = self._move(x, y, action)
        outs[(ix, iy)] = outs.get((ix, iy), 0.0) + (1.0 - slip)
        for pa in PERP[action]:
            px, py = self._move(x, y, pa)
            outs[(px, py)] = outs.get((px, py), 0.0) + slip / 2.0
        return [(p, cx, cy) for (cx, cy), p in outs.items()]

    def solve(self, reward_map, terminal_mask, slip_map):
        """
        reward_map    : (size,size) reward received upon ARRIVING in a cell
        terminal_mask : (size,size) bool, absorbing states
        slip_map      : (size,size) per-cell slip probability when leaving
        returns (V, policy)
        """
        size = self.size
        V = np.zeros((size, size), dtype=float)
        policy = np.empty((size, size), dtype=object)
        policy[:] = "UP"

        for _ in range(VI_ITERATIONS):
            newV = np.copy(V)
            delta = 0.0
            for x in range(size):
                for y in range(size):
                    if terminal_mask[x, y]:
                        newV[x, y] = 0.0
                        continue
                    slip = slip_map[x, y]
                    best_v, best_a = -1e18, "UP"
                    for a in ACTIONS:
                        q = 0.0
                        for p, nx, ny in self._outcomes(x, y, a, slip):
                            q += p * (reward_map[nx, ny] + GAMMA * V[nx, ny])
                        if q > best_v:
                            best_v, best_a = q, a
                    newV[x, y] = best_v
                    policy[x, y] = best_a
                    delta = max(delta, abs(newV[x, y] - V[x, y]))
            V = newV
            if delta < VI_TOLERANCE:
                break
        return V, policy
