"""
Configuration for the Autonomous Volcanic Exploration MDP system.

All tunable knobs live here so the behaviour (caution vs. speed,
how much it explores, how noisy the terrain is) can be changed in
one place.
"""

GRID_SIZE = 10
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

# ---- MDP solver ----
GAMMA = 0.95            # discount factor (favours long-horizon safe routes)
VI_ITERATIONS = 100     # max value-iteration sweeps per re-plan
VI_TOLERANCE = 1e-4     # early-stop when the value function stops changing

# ---- Uncertainty 1: stochastic motion (unstable ground / venting / wind) ----
# The agent INTENDS a direction but the terrain can push it sideways.
BASE_SLIP = 0.15        # prob. of slipping perpendicular to the intended move
GAS_SLIP_BONUS = 0.25   # extra slip while standing in a gas plume (lost control)

# ---- Uncertainty 2: partial observability ----
# The agent does not know the map in advance. It only perceives the true
# terrain within this Chebyshev radius and re-plans as it learns.
SENSOR_RADIUS = 1

# ---- Reward function (the agent's objective) ----
REWARDS = {
    "goal":   100,   # mission objective reached  (terminal, success)
    "crater": -100,  # fell into a crater         (terminal, failure)
    "lava":   -30,   # severe heat damage
    "gas":    -10,   # toxic plume, degrades control
    "step":   -1,    # time / energy cost per move  -> efficiency pressure
    "explore": 8,    # bonus for first visit to a new safe cell -> coverage
}

# ---- Ground-truth hazard counts placed on the hidden map ----
HAZARD_COUNTS = {"lava": 12, "gas": 10, "crater": 8}

# ---- Prior hazard probabilities for cells the agent has NOT yet sensed ----
# Used to compute EXPECTED penalties during planning, so the policy is
# cautious about the unknown instead of blindly optimistic.
_TOTAL = GRID_SIZE * GRID_SIZE
HAZARD_PRIOR = {
    "lava":   HAZARD_COUNTS["lava"]   / _TOTAL,
    "gas":    HAZARD_COUNTS["gas"]    / _TOTAL,
    "crater": HAZARD_COUNTS["crater"] / _TOTAL,
}

START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
MAX_STEPS = 200
