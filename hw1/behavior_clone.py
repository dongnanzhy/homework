#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python behavior_clone.py --expert_data_path expert_data/Humanoid-v2_20.pkl --envname Humanoid-v2 --render --num_rollouts 20
"""

import os
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import argparse
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--expert_data_path', type=str)
parser.add_argument('--envname', type=str)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=20,
                    help='Number of expert roll outs')
args = parser.parse_args()


def load_expert(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
    return data


def build_model(input_dim, output_dim, policy_name):
    model = keras.Sequential(name=policy_name)
    # Adds a densely-connected layer with 64 units to the model:
    model.add(keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='relu', batch_input_shape=(None, input_dim)))
    # Add another:
    model.add(keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='relu'))
    # Add another:
    model.add(keras.layers.Dense(64, kernel_initializer='lecun_normal', activation='relu'))
    # Add a linear layer with 1 output units:
    model.add(keras.layers.Dense(output_dim, activation='linear'))
    # Configure a model for mean-squared error regression.
    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                  loss='mse',  # mean squared logarithmic error
                  metrics=['mae'])  # mean absolute error
    return model


def train(obs, actions, model_name, epochs=60, batch_size=128):
    data, val_data, labels, val_labels = \
        train_test_split(obs, actions, test_size=0.2, random_state=42)

    model = build_model(obs.shape[1], actions.shape[1], model_name)

    model.fit(data, labels, epochs=epochs, batch_size=batch_size,
              validation_data=(val_data, val_labels))

    model.save_weights(os.path.join('models',
                                    '{0}_behavior_cloning.h5'.format(model.name)),
                       save_format='h5')


def run(model):
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.predict(obs[None,:])
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))


if __name__ == '__main__':
    expert_data = load_expert(args.expert_data_path)
    obs = expert_data['observations']
    actions = np.squeeze(expert_data['actions'], axis=1)

    _, policy_name = os.path.split(args.expert_data_path)
    model_name = policy_name.split('.')[0]
    model_path = os.path.join('models', '{0}_behavior_cloning.h5'.format(model_name))

    input_dim, output_dim = obs.shape[1], actions.shape[1]

    if not os.path.exists(model_path):
        train(obs, actions, model_name)
    model = build_model(input_dim, output_dim, model_name)
    model.load_weights(model_path)
    run(model)

