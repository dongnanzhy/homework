#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python dagger.py --expert_policy_path experts/Humanoid-v2.pkl --envname Humanoid-v2 --render --num_rollouts 20
"""

import os
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import argparse
import tf_util
import load_policy
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--expert_policy_path', type=str)
parser.add_argument('--envname', type=str)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=20,
                    help='Number of expert roll outs')
args = parser.parse_args()


def build_model(input_dim, output_dim, policy_name):
    model = keras.Sequential(name=policy_name)
    # Adds a densely-connected layer with 64 units to the model:
    model.add(keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='relu', batch_input_shape=(None, input_dim)))
    # Add another:
    model.add(keras.layers.Dense(128, kernel_initializer='lecun_normal', activation='relu'))
    # Add another:
    model.add(keras.layers.Dense(64, kernel_initializer='lecun_normal', activation='relu'))
    # Add a linear layer with 1 output units:
    model.add(keras.layers.Dense(output_dim, kernel_initializer='lecun_normal', activation='linear'))
    # Configure a model for mean-squared error regression.
    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                  loss='mse',  # mean squared logarithmic error
                  metrics=['mae'])  # mean absolute error
    return model


def train(expert_policy_path, model_name, dagger_iter=20, train_epochs=60, batch_size=128):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_path)
    print('loaded and built')
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    model, observations, actions = None, None, None

    for _ in range(dagger_iter):
        with tf.Session():
            tf_util.initialize()
            expert_observations = []
            expert_actions = []
            returns = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                steps = 0
                totalr = 0.
                while not done:
                    expert_action = policy_fn(obs[None, :])
                    expert_observations.append(obs)
                    expert_actions.append(expert_action)

                    if model is None:
                        action = expert_action
                    else:
                        action = model.predict(obs[None,:])
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)
            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

            expert_observations = np.array(expert_observations)
            expert_actions = np.squeeze(np.array(expert_actions), axis=1)

        if model is None:
            observations = expert_observations
            actions = expert_actions
            model = build_model(observations.shape[1], actions.shape[1], model_name)
        else:
            observations = np.concatenate((observations, expert_observations), axis=0)
            actions = np.concatenate((actions, expert_actions), axis=0)
            model = build_model(observations.shape[1], actions.shape[1], model_name)
            model.load_weights(os.path.join('models', '{0}_dagger.h5'.format(model_name)))

        data, val_data, labels, val_labels = \
            train_test_split(observations, actions, test_size=0.2, random_state=42)

        model.fit(data, labels, epochs=train_epochs, batch_size=batch_size,
                  validation_data=(val_data, val_labels))

        model.save_weights(os.path.join('models',
                                        '{0}_dagger.h5'.format(model.name)),
                           save_format='h5')

    return model


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
    model_name = '{0}_{1}'.format(args.envname, args.num_rollouts)
    model_path = os.path.join('models', '{0}_dagger.h5'.format(model_name))

    model = train(args.expert_policy_path, model_name)
    run(model)

