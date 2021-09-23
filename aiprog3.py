import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot

def main():
    grid_size = 10
    episodes = 5000
    steps = 200
    learning_rate = 0.2
    discount = 0.9
    epsilon = 0.1
    epsilon_delta = 0.01
    move_tax = 0
    qmatrix, training_rewards = train(episodes, steps, grid_size, epsilon, epsilon_delta, learning_rate, move_tax, discount)
    epsilon_delta = 0.00
    # use training qmatrix
    test_rewards = test(episodes, steps, qmatrix, grid_size, move_tax)
    result = matplotlib.pyplot.plot(range(test_rewards.size), test_rewards.tolist(), label="Test Output")
    matplotlib.pyplot.title("Reward Plot")
    matplotlib.pyplot.ylabel("Reward")
    matplotlib.pyplot.xlabel("Episode")
    legend = []
    legend.append("Test Result")
    result = matplotlib.pyplot.plot(range(training_rewards.size), training_rewards.tolist(), label="Training Output")
    legend.append("Training Result")
    matplotlib.pyplot.legend(legend)
    matplotlib.pyplot.draw()
    print("Test-Average: %d"%np.average(test_rewards))
    print("Test-Standard-Deviation: %d"%np.std(test_rewards))
    matplotlib.pyplot.show()

def train(episodes, number_of_steps, grid_size, epsilon, epsilon_delta,
                           learning_rate, move_tax, discount):
    qmatrix = Q_Matrix(epsilon, learning_rate, discount)
    best_qmatrix = qmatrix
    rewards = np.zeros((episodes,))
    for episode in range(1, episodes+1):
        # reduce epsilon
        if episode % 50 == 0 and epsilon > epsilon_delta:
            qmatrix.epsilon -= epsilon_delta
        grid = Grid(grid_size)
        total_reward = 0.0
        for step in range(number_of_steps):
            current_state = grid.get_state()
            move = qmatrix.pick_move(current_state)
            reward = grid.do_move(move, move_tax)
            new_state = grid.get_state()
            qmatrix.update_qmatrix(current_state, move, new_state, reward)
            total_reward += reward
        reward = float(total_reward)
        if reward > np.max(rewards):
            best_qmatrix = qmatrix
        rewards[episode - 1] = reward
    return best_qmatrix, rewards

def test(episodes, number_of_steps, qmatrix, grid_size, move_tax):
    rewards = np.zeros((episodes,))
    for episode in range(1, episodes + 1):
        grid = Grid(grid_size)
        total_reward = 0.0
        for step in range(number_of_steps):
            current_state = grid.get_state()
            move = qmatrix.pick_move(current_state)
            total_reward += grid.do_move(move, move_tax)
        rewards[episode - 1] = total_reward
    return rewards

class Grid:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = self.get_initial_grid()
        self.pos = self.get_pos()

    def get_initial_grid(self):
        # create grid with random cans, 0 for empty, 1 for can
        grid = np.random.randint(2, size=(self.grid_size, self.grid_size), dtype='int8')
        # random start location
        grid[np.random.randint(self.grid_size), np.random.randint(self.grid_size)] += 3
        # pad walls with 2 to denote wall
        return np.pad(array=grid, pad_width=1, mode='constant', constant_values=2)

    def get_pos(self):
        # get the coordinates where there is just a robot or robot and can
        coords = np.argwhere(self.grid >= 3)
        return coords.reshape(2, )

    def move_bot(self, next_cell):
        # decrease by robot value
        self.grid[tuple(self.pos)] -= 3
        # increase by robot value
        self.grid[tuple(next_cell)] += 3
        # set position
        self.pos = next_cell

    def get_next_square(self, move):
        # stay and pick up can
        if move == 0:
            move = (0,0)
        # up
        elif move == 1:
            move = (-1,0)
        # down
        elif move == 2:
            move = (1,0)
        # right
        elif move == 3:
            move = (0,1)
        # left
        elif move == 4:
            move = (0,-1)
        return self.pos + move

    def do_move(self, move, move_tax):
        reward = move_tax
        pos = self.get_pos()
        next_square = self.get_next_square(move)
        next_value = self.grid[tuple(next_square)]
        # staying put
        if move == 0:
            # can detected
            if next_value == 4:
                # pick up can
                reward += 10
                # remove can
                self.grid[tuple(pos)] -= 1
            else:
                # penalty for empty cell pickup
                reward -= 1
        else:
            # wall denoted by 2
            if next_value == 2:
                # creach into wall
                reward -= 5
            else:
                # move to open space
                self.move_bot(next_square)
        return reward

    def get_state(self):
        row, col = self.get_pos()
        # get surring suare coordinates
        surroundings = self.grid[row - 1:row + 2, col - 1:col + 2]
        # return values from this state
        state = tuple([surroundings[1, 1] - 3, surroundings[0, 1], surroundings[2, 1], surroundings[1, 2], surroundings[1, 0]])
        return state

class Q_Matrix:
    def __init__(self, epsilon, learning_rate, discount):
        # q matrix for each move, for each value
        self.qmatrix = pd.DataFrame({key: [0.0] * 5 for key in itertools.product([0,1,2], repeat=5)}, index=[0,1,2,3,4]).T
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount

    def pick_move(self, state):
        # chance of epsilon greedy, depending on decaying epsilon
        if np.random.sample() > self.epsilon:
            return np.argmax(self.qmatrix.loc[state].values)
        else:
            # else pick a random move
            return np.random.randint(5)

    def update_qmatrix(self, state, move, new_state, reward):
        old_qvalue = self.qmatrix.loc[state, move]
        new_qvalue = old_qvalue + self.learning_rate * (reward + self.discount *
                                                        np.max(self.qmatrix.loc[new_state].values) - old_qvalue)
        self.qmatrix.loc[state, move] = new_qvalue

if __name__ == "__main__":
    main()