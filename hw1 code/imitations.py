# Copyright (c) 2020 Yanyu Zhang zhangya@bu.edu All rights reserved.
import os
import numpy as np
import gym
from pyglet.window import key

import pandas as pd     # added by [hrsun]
from PIL import Image


def load_imitations(data_folder):
    """
    Given the folder containing the expert imitations, the data gets loaded and
    stored it in two lists: observations and actions.

    Parameter:
        data_folder: python string, the path to the folder containing the
                    actions csv files & observations jpg files
    return:
        observations:   python list of N numpy.ndarrays of size (160, 320, 3)
        actions:        python list of N numpy.ndarrays of size 3
        (N = number of (observation, action) - pairs)
    """
    # get actions
    csv_file = pd.read_csv(data_folder+'robot_log.csv', sep=';',header=None)
    csv_arr = csv_file.values
    actions = np.asarray(csv_arr[:, 1:4])
    print("actions[0]: ", actions[0], "; action[1]:", actions[1])

    # get observations
    obs_files = os.listdir(data_folder + '/IMG/')
    observations = [0]*int(len(obs_files))  # create list
    index = 0
    for filename in obs_files:  # loop through all files
        open_file_name = os.path.join(os.path.join(data_folder + '/IMG/'), filename)
        observations[index] = np.asarray(Image.open(open_file_name))
        index += 1
    observations = np.asarray(observations)
    print("observations[0]: ", observations[0])
    # print("observations[0] shape: ", observations[0].shape)

    return observations, actions

def save_imitations(data_folder, actions, observations):
    """
    1.1 f)
    Save the lists actions and observations in numpy .npy files that can be read
    by the function load_imitations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    for i in range(len(actions)):
        np.save(data_folder + '/action_' + '%05d' % i + '.npy', actions[i])
        np.save(data_folder + '/observation_' + '%05d' % i + '.npy', observations[i])
    print("====================================================")
    print("============ "+str(i+1)+" pairs of data saved ===============")
    print("====================================================")


class ControlStatus:
    """
    Class to keep track of key presses while recording imitations.
    """
    def __init__(self):
        self.stop = False
        self.save = False
        self.quit = False
        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

    def key_press(self, k, mod):
        if k == key.ESCAPE: self.quit = True
        if k == key.SPACE: self.stop = True
        if k == key.TAB: self.save = True
        if k == key.LEFT: self.steer = -1.0
        if k == key.RIGHT: self.steer = +1.0
        if k == key.UP: self.accelerate = +1.0 # changed
        if k == key.DOWN: self.brake = +0.4 # changed

    def key_release(self, k, mod):
        if k == key.LEFT and self.steer < 0.0: self.steer = 0.0
        if k == key.RIGHT and self.steer > 0.0: self.steer = 0.0
        if k == key.UP: self.accelerate = 0.0
        if k == key.DOWN: self.brake = 0.0


def record_imitations(imitations_folder):
    """
    Function to record own imitations by driving the car in the gym car-racing
    environment.
    imitations_folder:  python string, the path to where the recorded imitations
                        are to be saved

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    """
    env = gym.make('CarRacing-v0').env
    status = ControlStatus()
    total_reward = 0.0

    while not status.quit:
        observations = []
        actions = []
        # get an observation from the environment
        observation = env.reset()
        env.render()

        # set the functions to be called on key press and key release
        env.viewer.window.on_key_press = status.key_press
        env.viewer.window.on_key_release = status.key_release

        while not status.stop and not status.save and not status.quit:
            # collect all observations and actions
            observations.append(observation.copy())
            actions.append(np.array([status.steer, status.accelerate,
                                     status.brake]))
            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, info = env.step([status.steer,
                                                        status.accelerate,
                                                        status.brake])
            total_reward += reward
            env.render()

        if status.save:
            save_imitations(imitations_folder, actions, observations)
            status.save = False

        status.stop = False
        env.close()


# following code is for testing purpose only, need to be commented out later
data_folder = '/Users/hairuosun/Library/Mobile Documents/com~apple~CloudDocs/BU/Fall 2020/Courses/EC 500 A2/Project/Github Simulation/EC500_project/data/'
load_imitations(data_folder)
