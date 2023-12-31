import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import random
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import os
import time
import seaborn as sns
import pandas as pd

def plot_q_table(q_table, episode):
    plt.figure(figsize=(8, 12))
    plt.title(f"Q-table at Episode {episode}")

    df_q_table = pd.DataFrame(q_table)

    sns.heatmap(df_q_table, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.show()

def qtable_directions_map(qtable):
    qtable_val_max = qtable.max(axis=1).reshape(4, 12)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(4, 12)
    directions = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    qtable_directions = np.empty(qtable_best_action.shape, dtype=object) 

    for i in range(qtable_best_action.shape[0]):
        for j in range(qtable_best_action.shape[1]):
            action = qtable_best_action[i, j]
            if qtable_val_max[i, j] < 0: 
                qtable_directions[i, j] = directions[action]
            else:
                qtable_directions[i, j] = "" 

    return qtable_val_max, qtable_directions



def plot_q_values_map(qtable, env):
    qtable_val_max, qtable_directions = qtable_directions_map(qtable)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 3))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")


    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Greens", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    plt.show()


env = gym.make("CliffWalking-v0", render_mode="rgb_array")
trigger = lambda t: t % 1000 == 0 and t != 0
env = RecordVideo(env, video_folder="./save_videos1", episode_trigger=trigger, disable_logger=False)


# Define the Q-table dimensions
n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))


learning_rate = 0.01
discount_factor = 0.9

#Para hacerlo greedy, poner un epsilon muy bajo
epsilon = 1  # Exploration rate
num_episodes = 1001

# Q-learning algorithm
for episode in range(num_episodes):
    raw_state = env.reset()
    state = raw_state[0]

    done = False

    #epsilon = epsilon - 1/num_episodes #Reduce el epsilo a medida que avanza
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        # Execute action and observe new state
        new_state, reward, terminated, truncated, info = env.step(action)

        done = terminated

        # Guardia necesaria para que termine si gira hacia el risco
        if reward == -100:
            terminated = True
            done = True

        # Q-learning update
        old_value = q_table[state, action]
        next_max = np.max(q_table[new_state])

        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        q_table[state, action] = new_value


        state = new_sta

    
plot_q_table(q_table, episode)
plot_q_values_map(q_table, env)
env.close()
len(os.listdir("./save_videos1"))





