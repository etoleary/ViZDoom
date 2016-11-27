#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
import pickle
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import trange
from collections import deque

testing_scores_mean = []
learning_scores_mean = []

# Q-learning settings
learning_rate = 0.00025
#learning_rate = 0.00001
discount_factor = 1
#epochs = 100
epochs = 3


#learning_steps_per_epoch = 100000
#learning_steps_per_epoch = 10000
learning_steps_per_epoch = 500

replay_memory_size = 10000
last_four_states = deque(maxlen=4)
# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 50

# Other parameters
frame_repeat = 4 #maybe test 10
resolution = (45, 120)
episodes_to_watch = 10

model_savefile = "/tmp/model4.ckpt"
save_model = True
load_model = False
skip_learning = False
# Configuration file path
config_file_path = "../../examples/config/health_gathering.cfg"
output_file_path = "output4.txt"
output_file = open(output_file_path, 'w+')


# config_file_path = "../../examples/config/rocket_basic.cfg"
# config_file_path = "../../examples/config/basic.cfg"

# Converts and down-samples the input image
def preprocess(img):
    #print ("1 len: ", len(img), " len: ", len(img[0]))
    #print ("Image1: ", img)
    img = skimage.transform.resize(img, resolution)
    #print ("2 len: ", len(img), " len: ", len(img[0]))
    #print ("Image2: ", img)
    img = img.astype(np.float32)
    #print ("3 len: ", len(img), " len: ", len(img[0]))
    #print ("Image3: ", img)



    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 3
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
	#print("shape shape: ", self.s1[self.pos, :, :, :].shape)
        #print("shape: ",len(s1))
        self.s1[self.pos, :, :, :] = s1[0]
        self.a[self.pos] = action
        #print("s2 info: ")

        if not isterminal:
            #print(len(s2))
            #print(len(s2[0]))
            self.s2[self.pos, :, :, :] = s2[0]
            #print(self.s2.shape)
            #print(len(self.s2))
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get_last_N(self, n):
        #print("4 actions go here", np.take(self.a, range(self.pos - n, self.pos)))
        return np.take(self.s1, range(self.pos - n, self.pos)), np.take(self.a, range(self.pos - n, self.pos)),  np.take(self.s2, range(self.pos - n, self.pos)),  np.take(self.isterminal, range(self.pos - n, self.pos)),  np.take(self.r, range(self.pos - n, self.pos))
     
    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


def create_network(session, available_actions_count):
    # Create the input variables
    s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [3], name="State")
    a_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=32, kernel_size=[7, 7], stride=[4, 4],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))

    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=32, kernel_size=[5, 5], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))

    conv3 = tf.contrib.layers.convolution2d(conv2, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))

    conv3_flat = tf.contrib.layers.flatten(conv3) + game.get_state().game_variables[0]
    #print("Conv: ", conv3_flat)
	#add health to conv3_flat
    fc1 = tf.contrib.layers.fully_connected(conv3_flat, num_outputs=1024, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

    q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
    best_a = tf.argmax(q, 1)

    loss = tf.contrib.losses.mean_squared_error(q, target_q_)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)

    def function_learn(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q}
        l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={s1_: state})

    def function_simple_get_best_action(state):
	#print ("len-1: ", state.shape)	
        s1 = []
        return function_get_best_action(state.reshape([4, resolution[0], resolution[1], 3]))[0]


    return function_learn, function_get_q_values, function_simple_get_best_action


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:

        s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        #print("s1 shape: ", s1.shape)
        #print("action ", a)
        #if game.get_state().game_variables:
        #    r += game.get_state().game_variables[0]
        

        q2 = np.max(get_q_values(s2), axis=1)
        target_q = get_q_values(s1)
        
        #print("TargetQ shape: ", target_q.shape)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        learn(s1, target_q)


def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.004 * epochs  # 10% of learning time
        eps_decay_epochs = 0.104 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps
    #print("pre s1")
    #for thing in game.get_state().screen_buffer:
     #   print (thing[0])

    #print ("last 3 things", memory.get_last_N(3))
    if len(last_four_states) < 4:
        last_four_states.appendleft(preprocess(game.get_state().screen_buffer))
        last_four_states.appendleft(preprocess(game.get_state().screen_buffer))
        last_four_states.appendleft(preprocess(game.get_state().screen_buffer))
        last_four_states.appendleft(preprocess(game.get_state().screen_buffer))

    #s1 = preprocess(game.get_state().screen_buffer) # + last 4 states mayber
    s1 = last_four_states
    #print("lengths in order of depth")
    #print (len(s1))    
    #print (len(s1[0]))    
    #print (len(s1[0][0]))
    #print (len(s1[0][0][0]))
    #print("pre s1")
    #for thing in s1:
     #   print (thing[0])
    #print("shape: ",s1.shape)
    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(np.array(s1))
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    last_four_states.appendleft(preprocess(game.get_state().screen_buffer)) if not isterminal else None
    s2 = last_four_states if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    print("Initializing doom...", file=output_file)
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    print("Doom initialized.", file=output_file)
    return game


if __name__ == '__main__':
    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    last_four_states.append(preprocess(game.get_state().screen_buffer))
    last_four_states.append(preprocess(game.get_state().screen_buffer))
    last_four_states.append(preprocess(game.get_state().screen_buffer))
    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    #actions = [[0,0,0],[0,0,1],[0,1,0]]
    #print ("Actions: ", actions)    
    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    session = tf.Session()
    learn, get_q_values, get_best_action = create_network(session, len(actions))
    saver = tf.train.Saver()
    if load_model:
        print("Loading model from: ", model_savefile)
        saver.restore(session, model_savefile)
    else:
        init = tf.initialize_all_variables()
        session.run(init)
    print("Starting the training!")
    print("Starting the training!", output_file)

    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            print("\nEpoch %d\n-------" % (epoch + 1), file=output_file)
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            print("Training...\n", file=output_file)
            game.new_episode()
            for learning_step in trange(learning_steps_per_epoch):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    last_four_steps = []
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)
            print("%d training episodes played.\n" % train_episodes_finished, file=output_file)
            print ("scores: ", train_scores)
            print ("scores: ", train_scores,file=output_file)

            train_scores = np.array(train_scores)
            learning_scores_mean.append(train_scores.mean())
            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
            print("\nResults: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f,\n" % train_scores.max(), file=output_file)

            print("\nTesting...")
            print("\nTesting...\n",file=output_file)
            test_episode = []
            test_scores = []

            for test_episode in trange(test_episodes_per_epoch):
                game.new_episode()
                last_four_states.append(preprocess(game.get_state().screen_buffer))
                last_four_states.append(preprocess(game.get_state().screen_buffer))
                last_four_states.append(preprocess(game.get_state().screen_buffer))
                while not game.is_episode_finished():
                    last_four_states.appendleft(preprocess(game.get_state().screen_buffer))
                    state = last_four_states
                    best_action_index = get_best_action(np.array(state))

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)


            test_scores = np.array(test_scores)
            testing_scores_mean.append(test_scores.mean())
            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())
            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f\n" % test_scores.max(), file=output_file)

            print("Saving the network weigths to:", model_savefile)
            print("Saving the network weigths to:\n", model_savefile, file=output_file)

            saver.save(session, model_savefile)
            # pickle.dump(get_all_param_values(net), open('weights.dump', "wb"))

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))
            print("Total elapsed time: %.2f minutes\n" % ((time() - time_start) / 60.0), file=output_file)
    
    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")
    plt.plot(learning_scores_mean)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title('Scored vs. Episode')
    plt.savefig('graphs/healthgathering_learning_baseline_mean.png')
    plt.clf()
    plt.plot(testing_scores_mean)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title('Scored vs. Episode')
    plt.savefig('graphs/healthgathering_testing_baseline_mean.png')
    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        last_four_states.append(preprocess(game.get_state().screen_buffer))
        last_four_states.append(preprocess(game.get_state().screen_buffer))
        last_four_states.append(preprocess(game.get_state().screen_buffer))
        while not game.is_episode_finished():
            last_four_states.appendleft(preprocess(game.get_state().screen_buffer))
            state = last_four_states
            best_action_index = get_best_action(np.array(state))

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
