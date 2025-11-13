import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from forest_forage_env_assignment2 import ForestForageEnv
from q_learning_forest import train_q_learning

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clean_q_table_for_presentation(q_table, grid_size):
    cleaned_q_table = q_table.copy()
    for x in range(grid_size):
        for y in range(grid_size):
            for action in range(4):  # 0=up, 1=down, 2=left, 3=right
                invalid = False
                if action == 0 and y == grid_size - 1:      # Up
                    invalid = True
                elif action == 1 and y == 0:                # Down
                    invalid = True
                elif action == 2 and x == 0:                # Left
                    invalid = True
                elif action == 3 and x == grid_size - 1:    # Right
                    invalid = True
                if invalid:
                    cleaned_q_table[x, y, action] = np.nan  # Mask impossible moves
    return cleaned_q_table

def main():
    # ==== CLEAN PARAMETER SETUP (replacing argparse) ====
    episodes = 5000             # Training episodes
    alpha = 0.1                 # Learning rate
    gamma = 0.99                # Discount factor
    epsilon = 1.0               # Initial exploration rate
    epsilon_min = 0.1           # Minimum exploration rate
    epsilon_decay = 0.999       # Decay rate for epsilon per episode
    save_dir = "results"        # Output directory

    np.random.seed(42)  # For reproducibility

    # 1) Ensure results directory exists
    ensure_dir(save_dir)

    # 2) Initialize the environment
    env = ForestForageEnv()
    print(f"Initialized ForestForageEnv with grid size {env.grid_size}")

    # 3) Train Q-learning agent
    q_table_path = os.path.join(save_dir, "q_table.npy")
    print(f"Training for {episodes} episodes...")
    q_table, rewards = train_q_learning(
        env,
        num_episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        q_table_path=q_table_path
    )

    # 4) Plot and save training rewards
    print("Plotting training reward curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    reward_plot_path = os.path.join(save_dir, "training_rewards.png")
    plt.savefig(reward_plot_path)
    print(f"Training reward plot saved to {reward_plot_path}")
    plt.close()

    # 5) Evaluation and rendering
    print("\nEvaluation: Running one greedy episode and rendering...")
    state, _ = env.reset()
    done = False
    last_state = None
    repeat_count = 0
    epsilon_eval = 0.05


    visited_cells = set()

    while not done:
        x, y = state

        if last_state == (x, y):
            repeat_count += 1
        else:
            repeat_count = 0
        last_state = (x, y)

        if repeat_count > 3:
            action = env.action_space.sample()
        else:
            if np.random.rand() < epsilon_eval:
                action = env.action_space.sample()
            else:
                q_vals = q_table[x, y]
                valid_actions = get_valid_actions(env, (x, y))
                masked_q_vals = np.where(valid_actions, q_vals, -np.inf)
                max_val = np.max(masked_q_vals)

# Handle case when all are invalid (should not happen, but for safety):
                if np.isneginf(max_val):
                    action = env.action_space.sample()
                else:
                    candidates = np.flatnonzero(masked_q_vals == max_val)
                    action = int(np.random.choice(candidates))

        state, reward, done, _, info = env.step(action)
        if tuple(state) in visited_cells:
            reward -= 0.05   # Small loop penalty
        visited_cells.add(tuple(state))

        env.render()

    input("Grid episode finished. Press Enter to close the grid window and view heatmaps.")
    env.close()

    # ✅ Clean Q-table for presentation clarity
    cleaned_q_table = clean_q_table_for_presentation(q_table, env.grid_size)

    # 6) Combined Q-value heatmaps
    print("Generating combined Q-value heatmaps with G/H overlays...")
    berry_positions = env.berry_positions
    trap_positions = env.trap_positions
    actions = ["Up", "Down", "Left", "Right"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)

    for idx, action_name in enumerate(actions):
        ax = axes[idx]
        sns.heatmap(
            q_table[:, :, idx],
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar=(idx == 3),
            cbar_kws={"label": "Q-value"} if idx == 3 else None,
            ax=ax
        )
        ax.set_title(f"Action: {action_name}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position" if idx == 0 else "")

        ax.invert_yaxis()

        for (gx, gy) in berry_positions:
            ax.text(gx + 0.5, gy + 0.5, "G", color="red", fontsize=14, fontweight="bold", ha="center", va="center")
        for (hx, hy) in trap_positions:
            ax.text(hx + 0.5, hy + 0.5, "H", color="black", fontsize=14, fontweight="bold", ha="center", va="center")

    plt.tight_layout()
    combined_heatmap_path = os.path.join(save_dir, "all_q_values.png")
    plt.savefig(combined_heatmap_path)
    print(f"Saved combined heatmap to {combined_heatmap_path}")

    plt.show()
    plt.close()
    print("All results saved. Assignment 2 complete.")

def get_valid_actions(env, state):
    """
    Returns a boolean mask of valid actions (Up, Down, Left, Right) for the current state.
    True = valid, False = invalid
    """
    x, y = state
    valid = [True] * 4
    grid_size = env.grid_size

    # Up
    if y >= grid_size - 1:
        valid[0] = False
    # Down
    if y <= 0:
        valid[1] = False
    # Left
    if x <= 0:
        valid[2] = False
    # Right
    if x >= grid_size - 1:
        valid[3] = False

    return np.array(valid)

if __name__ == "__main__":
    main()
