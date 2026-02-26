# MAIN FUNCTION FOR Q-LEARNING EXPERIMENT ON TAXI-V3 ENVIRONMENT
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import time

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
        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])

    def greedy_action(self, state):
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error


# Decode Taxi State (for visual explanation)
def decode_state(state):

    destination = state % 4
    state //= 4

    passenger_location = state % 5
    state //= 5

    taxi_col = state % 5
    taxi_row = state // 5

    locations = ["Red", "Green", "Yellow", "Blue", "In Taxi"]
    destinations = ["Red", "Green", "Yellow", "Blue"]

    return {
        "taxi_row": taxi_row,
        "taxi_col": taxi_col,
        "passenger_location": locations[passenger_location],
        "destination": destinations[destination]
    }


# TRAIN FUNCTION (Returns Agent + Stats)
def train_config(alpha, epsilon, episodes=10000):

    env = gym.make("Taxi-v3")

    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = QLearningAgent(state_size, action_size,
                           alpha=alpha,
                           epsilon=epsilon)

    returns = []
    steps_per_episode = []

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
        steps_per_episode.append(steps)

    env.close()
    return agent, returns, steps_per_episode

# MOVING AVERAGE
def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')


# VISUAL SIMULATION
# ============================================================

def simulate_best_agent(agent, episodes=3):

    env = gym.make("Taxi-v3", render_mode="human")

    print("\n--- VISUAL SIMULATION (GREEDY POLICY) ---\n")

    for ep in range(episodes):

        state, _ = env.reset()
        done = False
        total_reward = 0

        print(f"\nEpisode {ep+1}")

        decoded = decode_state(state)
        print("Initial State:", decoded)

        while not done:

            action = agent.greedy_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            decoded = decode_state(next_state)

            print("Taxi at:", (decoded["taxi_row"], decoded["taxi_col"]),
                  "| Passenger:", decoded["passenger_location"],
                  "| Destination:", decoded["destination"])

            total_reward += reward
            state = next_state

            time.sleep(0.4)

        if total_reward > 0:
            print("SUCCESS: Passenger Delivered ✅")
        else:
            print("FAILED ❌")

        print("Episode Return:", total_reward)
        time.sleep(1)

    env.close()


# MAIN EXPERIMENT
def main():

    results_dir = "results_q_learning"
    os.makedirs(results_dir, exist_ok=True)

    configs = {
        "baseline": (0.1, 0.1),
        "alpha_0.01": (0.01, 0.1),
        "alpha_0.001": (0.001, 0.1),
        "alpha_0.2": (0.2, 0.1),
        "eps_0.2": (0.1, 0.2),
        "eps_0.3": (0.1, 0.3)
    }

    summary = []
    best_agent = None
    best_return = -float("inf")

    plt.figure(figsize=(10,6))

    for name, (alpha, epsilon) in configs.items():

        print(f"\nTraining {name}...")
        agent, returns, steps = train_config(alpha, epsilon)

        avg_return = np.mean(returns[-100:])
        avg_steps = np.mean(steps[-100:])

        summary.append([name, alpha, epsilon,
                        round(avg_return,2),
                        round(avg_steps,2)])

        if avg_return > best_return:
            best_return = avg_return
            best_agent = agent

        ma_returns = moving_average(returns)
        plt.plot(ma_returns, label=f"{name} (avgR={avg_return:.2f})")

    plt.title("Taxi-v3 Q-Learning: Moving Avg Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "returns_moving_avg.png"))
    plt.close()

    # Save CSV
    with open(os.path.join(results_dir, "comparison_table.csv"),
              mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Config", "Alpha", "Epsilon",
                         "Avg Return (last100)",
                         "Avg Steps (last100)"])
        writer.writerows(summary)

    print("\nParameter Performance Comparison")
    print("-------------------------------------------------")
    for row in summary:
        print(row)

    print("\nBest configuration selected for visualization.")

 
    # VISUALIZE BEST LEARNED AGENT
    simulate_best_agent(best_agent, episodes=3)


if __name__ == "__main__":
    main()
