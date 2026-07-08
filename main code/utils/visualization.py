"""
Visualization: renders three panels and saves them to a PNG
(headless-safe via the Agg backend).

  Panel 1  Discovered terrain + the agent's actual path
  Panel 2  Final value-function heatmap with the optimal policy arrows
  Panel 3  Mission metrics
"""

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

from config import ACTION_DELTAS

TERRAIN_COLOR = {
    "safe": "#e8efe6",
    "lava": "#e8743b",
    "gas": "#f2d14e",
    "crater": "#8a1c1c",
    "goal": "#2e9e5b",
    None: "#3a3f4b",          # unexplored
}


def render(agent, env, out_path="mission_result.png"):
    size = env.size
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # ---------- Panel 1: discovered terrain + path ----------
    ax = axes[0]
    grid_rgb = np.zeros((size, size, 3))
    for x in range(size):
        for y in range(size):
            t = agent.known[x, y]
            c = TERRAIN_COLOR.get(t, TERRAIN_COLOR[None])
            grid_rgb[x, y] = matplotlib.colors.to_rgb(c)
    ax.imshow(grid_rgb, origin="upper")

    traj = np.array(agent.trajectory)
    ax.plot(traj[:, 1], traj[:, 0], color="#1f6fdb", linewidth=2.5,
            marker="o", markersize=4, label="path")
    ax.plot(traj[0, 1], traj[0, 0], "ws", markersize=12, mec="black", label="start")
    gx, gy = agent.goal
    ax.plot(gy, gx, "*", color="gold", markersize=22, mec="black", label="goal")
    if not agent.alive:
        ax.plot(traj[-1, 1], traj[-1, 0], "X", color="red", markersize=18,
                mec="black", label="destroyed")
    ax.set_title("Discovered terrain & path", fontsize=13, fontweight="bold")
    ax.set_xticks(range(size)); ax.set_yticks(range(size))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # ---------- Panel 2: value heatmap + policy ----------
    ax = axes[1]
    V = agent.last_V if agent.last_V is not None else np.zeros((size, size))
    im = ax.imshow(V, cmap="RdYlGn", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if agent.last_policy is not None:
        for x in range(size):
            for y in range(size):
                if agent.known[x, y] in ("crater", "goal"):
                    continue
                a = agent.last_policy[x, y]
                dx, dy = ACTION_DELTAS[a]
                ax.arrow(y, x, dy * 0.3, dx * 0.3, head_width=0.18,
                         head_length=0.18, fc="black", ec="black", alpha=0.6)
    ax.set_title("Value function & optimal policy", fontsize=13, fontweight="bold")
    ax.set_xticks(range(size)); ax.set_yticks(range(size))

    # ---------- Panel 3: metrics ----------
    ax = axes[2]; ax.axis("off")
    s = agent.summary()
    status = ("OBJECTIVE ACHIEVED" if s["reached_goal"]
              else ("DESTROYED" if not s["alive"] else "TIMED OUT"))
    color = "#2e9e5b" if s["reached_goal"] else "#8a1c1c"
    lines = [
        ("Mission status", status),
        ("Steps taken", s["steps"]),
        ("Total reward", s["total_reward"]),
        ("Cells explored", f"{s['cells_explored']} / {size*size}"),
        ("Coverage", f"{s['coverage_pct']} %"),
        ("Start -> Goal", f"{agent.trajectory[0]} -> {agent.goal}"),
    ]
    ax.text(0.5, 0.93, "MISSION SUMMARY", ha="center", fontsize=15,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.83, status, ha="center", fontsize=14, fontweight="bold",
            color=color, transform=ax.transAxes)
    y0 = 0.68
    for k, v in lines[1:]:
        ax.text(0.08, y0, f"{k}:", fontsize=12, transform=ax.transAxes)
        ax.text(0.92, y0, f"{v}", fontsize=12, ha="right",
                fontweight="bold", transform=ax.transAxes)
        y0 -= 0.11

    fig.suptitle("Autonomous Volcanic Exploration  —  MDP under Uncertainty",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
