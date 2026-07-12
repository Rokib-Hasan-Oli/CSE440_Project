# Autonomous Volcanic Exploration — MDP System Design

**Objective.** Navigate a simulated volcanic terrain, *managing uncertain
environmental conditions* (lava, craters, gas), so as to *maximize
exploration efficiency* while *keeping the agent safe*.

The core idea is to treat this as a **stochastic, partially-observed MDP
solved by receding-horizon Value Iteration**. The agent does not know the
map in advance; it senses locally, re-plans every step on its current
belief, and acts under motion that is itself uncertain.

---

## 1. MDP formulation

A Markov Decision Process is the tuple **(S, A, T, R, γ)**.

### States — S
Each grid cell `(x, y)` on the 10×10 terrain is a state. Two states are
**absorbing (terminal)**: the goal (success) and any crater the agent has
discovered (failure).

### Actions — A
`{UP, DOWN, LEFT, RIGHT}`.

### Transition model — T  *(uncertainty #1: motion)*
Movement is **stochastic**, modelling unstable ground, venting and wind:

```
P(intended cell)        = 1 − slip
P(each perpendicular)   = slip / 2
```

`slip = 0.15` normally, rising to `0.40` inside a gas plume (degraded
control). A move into a wall keeps the agent in place. Because the Bellman
backup averages over these outcomes, the optimal policy automatically keeps
a **safety margin** away from craters and lava — it "knows" it might slip.

### Reward function — R
| Event | Reward | Purpose |
|---|---|---|
| Reach goal | **+100** | mission objective |
| Fall in crater | **−100** | fatal — terminal failure |
| Enter lava | **−30** | severe hazard |
| Enter gas | **−10** | minor hazard + more slip |
| Each step | **−1** | time/energy cost → **efficiency** |
| First visit to a new safe cell | **+8** | **exploration** incentive |

The step cost pushes for efficiency; the exploration bonus rewards
covering new ground; the hazard penalties enforce safety. Their balance is
what the policy optimizes.

### Discount factor — γ
`γ = 0.95`. High enough that the agent plans long, safe routes rather than
greedy short ones.

### Solution — Value Iteration
```
V(s)  = max_a  Σ_s'  P(s'|s,a) · [ R(s') + γ·V(s') ]
π(s)  = argmax_a ( same )
```
Iterated to convergence (`Δ < 1e−4`). Terminal states have `V = 0`; their
reward is collected by the neighbour that transitions into them.

---

## 2. Handling uncertainty #2: the unknown map

The terrain is **partially observable** — the agent only sees the true type
of cells within a sensor radius of 1. We keep the problem an MDP (not a full
POMDP) using the standard practical technique of **planning on the current
belief and re-planning as it updates**:

* **Belief map.** Every cell is either *sensed* (true type known) or
  *unexplored*.
* **Expected-cost planning.** An unexplored cell is scored by its *expected*
  reward under a hazard prior:
  `R = step + P(crater)·(−100) + P(lava)·(−30) + P(gas)·(−10) + P(safe)·(+8)`.
  This makes the agent **cautious about the unknown** yet still **drawn to
  explore** likely-safe cells.
* **Receding horizon.** Each step: **sense → re-plan (Value Iteration) →
  act**. Newly revealed craters become terminal states the policy routes
  around on the very next plan.

This loop is what turns a static path-finder into an *autonomous explorer*.

---

## 3. Safety mechanisms

1. **Stochastic backups** keep the policy clear of hazard edges (it accounts
   for slipping in).
2. **Craters are terminal** with a large penalty, so any route through a
   known crater is strictly dominated.
3. **Expected-penalty planning** treats the unknown pessimistically enough
   to avoid reckless shortcuts, but not so much that the agent freezes.
4. **Gas raises the slip probability**, so the planner treats gas zones as
   both costly *and* harder to control, and tends to skirt them.

---

## 4. Exploration efficiency

* **Coverage** is driven by the +8 first-visit bonus: the agent sweeps
  reachable safe cells instead of beelining to the goal.
* **Efficiency** is enforced by the −1 step cost and discounting, so it
  won't chase distant bonuses that aren't worth the travel.
* The tradeoff is a single knob (`REWARDS["explore"]` in `config.py`):
  higher → more thorough but slower; lower → more direct.

---

## 5. Architecture

```
config.py                 all parameters (rewards, slip, priors, γ)
environment/grid_world.py hidden ground-truth terrain
environment/hazards.py    random hazard placement
mdp/mdp_solver.py         stochastic Value Iteration
agent/explorer.py         sense → plan → act → learn loop + metrics
utils/visualization.py    3-panel PNG (terrain+path, value+policy, metrics)
main.py                   run a mission (optional seed for reproducibility)
```

Run: `python main.py 7`  (the integer is an optional random seed).

---

## 6. Observed behaviour

Single seeded run: **goal reached in 38 steps, 74% of the terrain explored,
total reward 260**, after surviving a lava slip. Across 10 random maps:
**7 reached the goal, 2 were destroyed, 1 timed out, ~60% average coverage.**
The non-trivial failure rate is expected and honest — with dense craters,
local sensing and stochastic motion, perfect safety is not guaranteed; it
is the *expected* return that is optimized. Tuning the explore bonus,
sensor radius, or slip level shifts the safety/coverage/speed balance.

---

## 7. Possible extensions

* Full **POMDP** with a belief distribution over hazards (vs. point belief).
* **Constrained MDP / risk-sensitive** objective to bound crater probability
  explicitly rather than only penalizing it.
* **Information-gain** reward (expected map entropy reduction) for smarter
  frontier-style exploration.
* **Q-learning** to learn the policy from experience instead of planning.
