import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from forest_forage_env_assignment2 import ForestForageEnv

def get_valid_actions(env, state):
    """
    Returns a boolean mask of valid actions from the given state in the environment.
    Actions: 0=up, 1=down, 2=left, 3=right
    """
    x, y = state
    valid = [True, True, True, True]
    if y >= env.grid_size - 1:
        valid[0] = False  # up invalid
    if y <= 0:
        valid[1] = False  # down invalid
    if x <= 0:
        valid[2] = False  # left invalid
    if x >= env.grid_size - 1:
        valid[3] = False  # right invalid
    return np.array(valid)


def train_q_learning(
    env,
    num_episodes=1000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.00,
    epsilon_decay=0.995,
    q_table_path="q_table.npy"
):
    """
    Train a Q-learning agent on the given environment.

    Args:
        env: Gym environment instance
        num_episodes: number of training episodes
        alpha: learning rate
        gamma: discount factor
        epsilon: initial exploration rate
        epsilon_min: minimum exploration rate
        epsilon_decay: decay rate for epsilon after each episode
        q_table_path: file path to save the learned Q-table

    Returns:
        q_table: learned Q-table as a NumPy array of shape (grid_size, grid_size, n_actions)
        rewards_per_episode: list of total rewards collected per episode
    """
    # Initialize Q-table with zeros
    grid_size = env.grid_size
    n_actions = env.action_space.n
    q_table = np.zeros((grid_size, grid_size, n_actions))

    rewards_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        visited_cells = set() 

        while not done:
            x, y = state
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                valid_actions = get_valid_actions(env, (x, y))
                masked_q_vals = np.where(valid_actions, q_table[x, y], -np.inf)
                action = int(np.argmax(masked_q_vals))


            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            if tuple(state) in visited_cells:
                reward -= 0.05   # 👈 small loop penalty
            visited_cells.add(tuple(state))

            x2, y2 = next_state

            # Q-learning update
            best_next = np.max(q_table[x2, y2])
            td_target = reward + gamma * best_next
            td_error = td_target - q_table[x, y, action]
            q_table[x, y, action] += alpha * td_error

            state = next_state
            total_reward += reward

        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        # Optional: print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Total reward: {total_reward:.2f} | ε: {epsilon:.3f}")

    # Save Q-table
    np.save(q_table_path, q_table)
    print(f"Q-table saved to '{q_table_path}'")

    return q_table, rewards_per_episode


def plot_training_stats(rewards_per_episode):
    """
    Plot total reward per episode over training.
    """
    plt.figure()
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training: Total Reward per Episode")
    plt.show()


def plot_q_table(q_table):
    """
    Plot heatmaps of Q-values for each action using seaborn.
    """
    n_actions = q_table.shape[2]
    actions = ["Up", "Down", "Left", "Right"]
    for action in range(n_actions):
        plt.figure()
        sns.heatmap(
            q_table[:, :, action],
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={'label': 'Q-value'}
        )
        plt.title(f"Q-values for action: {actions[action]}")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.show()


if __name__ == "__main__":
    # Create the environment
    env = ForestForageEnv()

    # Hyperparameters
    NUM_EPISODES = 1000
    ALPHA = 0.1
    GAMMA = 0.99
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995

    # Train the agent
    q_table, rewards = train_q_learning(
        env,
        num_episodes=NUM_EPISODES,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        q_table_path="q_table.npy"
    )

    # Visualize
    plot_training_stats(rewards)
    plot_q_table(q_table)

    env.close()
