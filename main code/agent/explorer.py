"""
Autonomous exploration agent.

Loop (receding-horizon planning under partial observability):
    1. SENSE   reveal true terrain within SENSOR_RADIUS, update belief map
    2. PLAN    build reward/terminal/slip models from current belief,
               solve the stochastic MDP -> optimal policy
    3. ACT     take the policy's action; motion executes STOCHASTICALLY
               on the real terrain (it may slip)
    4. LEARN   collect real reward, mark cell visited, repeat

Unknown cells are scored by their EXPECTED reward under the hazard prior
plus an exploration bonus, so the agent is simultaneously (a) cautious
about the unknown and (b) drawn to cover new ground -- balancing safety
against exploration efficiency.
"""

import random
import numpy as np

from config import (REWARDS, HAZARD_PRIOR, BASE_SLIP, GAS_SLIP_BONUS,
                    SENSOR_RADIUS, START, GOAL, MAX_STEPS)
from mdp.mdp_solver import MDPSolver


class Explorer:
    def __init__(self, env, start=START, goal=GOAL):
        self.env = env
        self.size = env.size
        self.pos = start
        self.goal = goal
        self.solver = MDPSolver(self.size)

        # belief: None = unexplored, else the sensed true terrain type
        self.known = np.full((self.size, self.size), None, dtype=object)
        self.known[goal] = "goal"

        self.visited = set()
        self.sensed = set()

        # metrics
        self.total_reward = 0.0
        self.steps = 0
        self.alive = True
        self.reached_goal = False
        self.trajectory = [start]
        self.last_action = None
        self.last_V = None
        self.last_policy = None

        self.visited.add(start)
        self._sense()

    # ---------- 1. SENSE ----------
    def _sense(self):
        x, y = self.pos
        r = SENSOR_RADIUS
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    self.known[nx, ny] = self.env.get_type(nx, ny)
                    self.sensed.add((nx, ny))

    # ---------- 2. PLAN: turn the belief into MDP models ----------
    def _build_models(self):
        size = self.size
        reward_map = np.zeros((size, size), dtype=float)
        terminal = np.zeros((size, size), dtype=bool)
        slip = np.full((size, size), BASE_SLIP, dtype=float)

        p = HAZARD_PRIOR
        p_safe = max(0.0, 1.0 - (p["crater"] + p["lava"] + p["gas"]))

        for x in range(size):
            for y in range(size):
                if (x, y) == self.goal:
                    reward_map[x, y] = REWARDS["goal"]
                    terminal[x, y] = True
                    continue

                t = self.known[x, y]
                base = REWARDS["step"]

                if t is None:
                    # expected penalty over what it MIGHT be + lure to explore
                    exp_pen = (p["crater"] * REWARDS["crater"]
                               + p["lava"] * REWARDS["lava"]
                               + p["gas"] * REWARDS["gas"])
                    reward_map[x, y] = base + exp_pen + p_safe * REWARDS["explore"]
                elif t == "crater":
                    reward_map[x, y] = REWARDS["crater"]
                    terminal[x, y] = True
                elif t == "lava":
                    reward_map[x, y] = base + REWARDS["lava"]
                elif t == "gas":
                    reward_map[x, y] = base + REWARDS["gas"]
                    slip[x, y] = BASE_SLIP + GAS_SLIP_BONUS
                else:  # known safe
                    bonus = 0.0 if (x, y) in self.visited else REWARDS["explore"]
                    reward_map[x, y] = base + bonus
        return reward_map, terminal, slip

    # ---------- 3. ACT: stochastic execution on the real terrain ----------
    def _sample_move(self, action):
        x, y = self.pos
        true_t = self.env.get_type(x, y)
        slip = BASE_SLIP + (GAS_SLIP_BONUS if true_t == "gas" else 0.0)
        outs = self.solver._outcomes(x, y, action, slip)
        r = random.random()
        cum = 0.0
        for prob, nx, ny in outs:
            cum += prob
            if r <= cum:
                return nx, ny
        return outs[-1][1], outs[-1][2]

    def step(self):
        if not self.alive or self.reached_goal:
            return False

        reward_map, terminal, slip = self._build_models()
        V, policy = self.solver.solve(reward_map, terminal, slip)
        self.last_V, self.last_policy = V, policy

        action = policy[self.pos]
        self.last_action = action
        nx, ny = self._sample_move(action)

        self.pos = (nx, ny)
        self.trajectory.append(self.pos)
        self.steps += 1

        # real reward on the real terrain
        self.total_reward += REWARDS["step"]
        self._sense()
        t = self.env.get_type(nx, ny)
        if (nx, ny) == self.goal:
            self.total_reward += REWARDS["goal"]
            self.reached_goal = True
        elif t == "crater":
            self.total_reward += REWARDS["crater"]
            self.alive = False
        elif t == "lava":
            self.total_reward += REWARDS["lava"]
        elif t == "gas":
            self.total_reward += REWARDS["gas"]
        else:
            if (nx, ny) not in self.visited:
                self.total_reward += REWARDS["explore"]
        self.visited.add((nx, ny))
        return self.alive and not self.reached_goal

    # ---------- driver ----------
    def run(self, max_steps=MAX_STEPS, verbose=True):
        if verbose:
            self._banner()
        while self.steps < max_steps and self.alive and not self.reached_goal:
            self.step()
            if verbose:
                self._print_step()
        return self.summary()

    def coverage(self):
        return len(self.sensed) / (self.size * self.size)

    def summary(self):
        return {
            "steps": self.steps,
            "total_reward": round(self.total_reward, 1),
            "reached_goal": self.reached_goal,
            "alive": self.alive,
            "cells_explored": len(self.sensed),
            "coverage_pct": round(100 * self.coverage(), 1),
            "trajectory": self.trajectory,
        }

    # ---------- console output ----------
    def _banner(self):
        print("=" * 56)
        print("   Autonomous Volcanic Exploration  -  MDP Agent".center(56))
        print("=" * 56)
        print(f"Start {self.pos}  ->  Goal {self.goal}")
        print(f"Rewards: goal +{REWARDS['goal']}, crater {REWARDS['crater']}, "
              f"lava {REWARDS['lava']}, gas {REWARDS['gas']}, "
              f"step {REWARDS['step']}, explore +{REWARDS['explore']}")
        print(f"Slip prob {BASE_SLIP} (+{GAS_SLIP_BONUS} in gas), "
              f"sensor radius {SENSOR_RADIUS}")
        print("-" * 56)

    def _print_step(self):
        x, y = self.pos
        t = self.env.get_type(x, y)
        icon = {"safe": "  safe", "lava": "  LAVA", "gas": "   gas",
                "crater": "CRATER", "goal": "  GOAL"}[t]
        arrow = {"UP": "^", "DOWN": "v", "LEFT": "<", "RIGHT": ">"}[self.last_action]
        tag = ""
        if self.reached_goal:
            tag = "  <-- OBJECTIVE ACHIEVED"
        elif not self.alive:
            tag = "  <-- DESTROYED"
        print(f"Step {self.steps:>3} | act {arrow} | pos ({x},{y}) {icon} | "
              f"R={self.total_reward:>7.1f}{tag}")
