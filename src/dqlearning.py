import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.optimizers import Adam
import gymnasium as gym
from collections import deque
import flappy_bird_env
import cv2
import pygame

def build_model(input_shape, actions):
    model = Sequential()
    
    # Assuming input_shape is something like (800, 576, 3)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())  # Flatten the data for the dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    
    model.summary()
    return model

def build_agent(model, actions):
  print("Building Policy")
  policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.5, value_min=.0001, value_test=.0, nb_steps=6000)
  print("Allocating Memory")
  memory = SequentialMemory(limit=10000, window_length=1)
  print("Building DQN")
  dqn = DQNAgent(model=model, memory=memory, policy=policy, dueling_type='avg',nb_actions=actions, nb_steps_warmup=500)
  return dqn

#Create enviroment
env = gym.make("FlappyBird-v0", render_mode="rgb_array")
observation, info = env.reset()
actions = env.action_space.n

print(observation.shape)

env.render()
print("Building Model")
model = build_model(observation.shape, actions)
print("Building Agent")
dqn = build_agent(model, actions)

num_episodes = 1000

#Training the Neural Network
print("Compiling DQN")
dqn.compile(Adam(learning_rate=0.0025))

print("Training Network")
dqn.fit(env, nb_steps=50, visualize=False, verbose=2)

dqn.save_weights("try1.h5")

for episode in range(num_episodes):
    action = env.action_space  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(0)

    if terminated or truncated:
        observation, info = env.reset()

env.close()