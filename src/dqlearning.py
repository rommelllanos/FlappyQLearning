import flappy_bird_env
import gymnasium as gym
import numpy as np
import tensorflow as tf
#import keras
#from keras import layers
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2  # OpenCV for image processing
import pygame

def build_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

""" #preprocess state
def rgb_to_grayscale(rgb_image):
    # Ensure the image has three channels
    if rgb_image.shape[-1] != 3:
        raise ValueError("The input image must have three channels (RGB).")

    # Convert RGB to Grayscale using a weighted average to account for human perception
    grayscale_image = np.dot(rgb_image[...,:3], [0.2989, 0.5870, 0.1140])

    return grayscale_image """

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = build_model()

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

pygame.init()

env = gym.make("FlappyBird-v0", render_mode="human")
observation, info = env.reset()
state_shape = env.observation_space
action_size = env.action_space.n

print(action_size)
print(state_shape)
print(info)

env.render()

num_episodes = 1000

for episode in range(num_episodes):
    action = env.action_space  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(0)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

""" model = build_model(state_shape, action_size)
replay_buffer = deque(maxlen=2000)

epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32

num_episodes = 1000

for episode in range(num_episodes):
    state = rgb_to_grayscale(env.reset())
    state = np.reshape(state, [1, *state_shape])
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(state)[0])

        next_state, reward, done, _ = env.step(action)
        next_state = rgb_to_grayscale(next_state)
        next_state = np.reshape(next_state, [1, *state_shape])

        # Store in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(replay_buffer) > batch_size:
            minibatch = random.sample(replay_buffer, batch_size)
            for b_state, b_action, b_reward, b_next_state, b_done in minibatch:
                target = b_reward
                if not b_done:
                    target = b_reward + 0.99 * np.amax(model.predict(b_next_state)[0])
                target_f = model.predict(b_state)
                target_f[0][b_action] = target
                model.fit(b_state, target_f, epochs=1, verbose=0)

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode: {episode}, Total Reward: {total_reward}")

env.close()
 """