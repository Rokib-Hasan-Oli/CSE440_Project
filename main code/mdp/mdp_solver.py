import numpy as np
from config import ACTIONS, GAMMA  # actions list & discount factor

class MDPSolver:
    def __init__(self, env):
        self.env = env                              # environment (grid)
        self.V = np.zeros_like(env.grid)            # value function table (initialize 0)
        self.policy = np.zeros(env.grid.shape, dtype=int)  # best action for each cell

    def get_next_state(self, x, y, action):
        # define movement directions
        moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        dx, dy = moves[action]
        nx, ny = x + dx, y + dy  # compute next position

        # check if next position is valid
        if self.env.is_valid(nx, ny):
            return nx, ny
        return x, y  # stay if invalid move

    def value_iteration(self, iterations=100):
        # repeat updates multiple times
        for _ in range(iterations):
            new_V = np.copy(self.V)  # copy current values

            # loop through all grid cells
            for x in range(self.env.size):
                for y in range(self.env.size):
                    values = []  # store value for each action

                    # try all possible actions
                    for a in range(len(ACTIONS)):
                        nx, ny = self.get_next_state(x, y, a)  # next state
                        reward = self.env.get_cell(nx, ny)     # reward at next state

                        # Bellman update: reward + discounted future value
                        values.append(reward + GAMMA * self.V[nx][ny])

                    best_value = max(values)        # choose max value
                    best_action = np.argmax(values) # index of best action

                    new_V[x][y] = best_value        # update value function
                    self.policy[x][y] = best_action # update policy

            self.V = new_V  # replace old values

        return self.V, self.policy  # return final value & policy
