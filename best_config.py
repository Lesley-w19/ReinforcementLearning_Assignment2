# Re-running Using Best Configuration
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_size, action_size,
                 alpha=0.1,
                 gamma=0.9,
                 epsilon=0.1):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error



# Moving Average
def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Train Best Configuration
def train_best_config(episodes=10000):

    env = gym.make("Taxi-v3")

    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = QLearningAgent(
        state_size,
        action_size,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1
    )

    returns = []
    steps_list = []

    for ep in range(episodes):

        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward
            steps += 1

        returns.append(total_reward)
        steps_list.append(steps)

    env.close()
    return returns, steps_list

# MAIN
returns, steps = train_best_config()

ma_returns = moving_average(returns, window=100)
ma_steps = moving_average(steps, window=100)


# Plot 1: Returns
plt.figure()
plt.plot(returns, alpha=0.3, label="Episode Return")
plt.plot(range(99, len(returns)), ma_returns, label="Moving Average (100)")
plt.title("Returns - Best Config (α=0.1, ε=0.1)")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.savefig("./results_q_learning/best_config_returns.png")
plt.show()

# Plot 2: Steps per Episode
plt.figure()
plt.plot(steps, alpha=0.3, label="Steps per Episode")
plt.plot(range(99, len(steps)), ma_steps, label="Moving Average (100)")
plt.title("Steps per Episode - Best Config (α=0.1, ε=0.1)")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.savefig("./results_q_learning/best_config_steps.png") 
plt.show()