"""
Entry point: run one autonomous exploration mission.

Usage:
    python main.py            # random mission
    python main.py 7          # reproducible mission with seed 7
    python main.py 7 --quiet  # no per-step console output
"""

import sys
from environment.grid_world import GridWorld
from environment.hazards import populate_true_terrain
from agent.explorer import Explorer
from config import START, GOAL


def run_mission(seed=None, verbose=True, make_plot=True, out_path="mission_result.png"):
    env = GridWorld()
    populate_true_terrain(env, seed=seed)

    agent = Explorer(env, start=START, goal=GOAL)
    summary = agent.run(verbose=verbose)

    if verbose:
        print("-" * 56)
        print("RESULT:", "SUCCESS" if summary["reached_goal"]
              else ("FAILED (destroyed)" if not summary["alive"] else "TIMED OUT"))
        print(f"  steps={summary['steps']}  reward={summary['total_reward']}  "
              f"coverage={summary['coverage_pct']}%")

    if make_plot:
        from utils.visualization import render
        path = render(agent, env, out_path=out_path)
        if verbose:
            print(f"  visualization saved -> {path}")

    return agent, summary


if __name__ == "__main__":
    seed = None
    verbose = True
    for arg in sys.argv[1:]:
        if arg == "--quiet":
            verbose = False
        else:
            try:
                seed = int(arg)
            except ValueError:
                pass
    run_mission(seed=seed, verbose=verbose)
