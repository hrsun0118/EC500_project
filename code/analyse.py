# Copyright (c) 2020 Yanyu Zhang zhangya@bu.edu All rights reserved.
import numpy as np
import matplotlib.pyplot as plt
from imitations import load_imitations


def plot_acc(data1, name):
    acc1 = np.load(data1, allow_pickle = True)  # why "allow_pickle = True"
                                                # (online: “Pickling” is the process whereby
                                                # a Python object hierarchy is converted into a byte stream)
    plt.figure()
    plt.plot(range(len(acc1)), acc1, label=name)

    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    # plt.show()
    plt.savefig('images/'+str(name)+'.png')

# plot_acc('result/easy_gray.npy', 'MyNet')     # do we only need to put easy_gray.npy file in the "result" folder?

def data_analyse_seven(data_folder):
    observations, actions = load_imitations(data_folder)
    left, left_throttle, right, right_throttle, acc, brakes, keep = 0,0,0,0,0,0,0
    for action in actions: # [steer, throttle, brake]
        if action[0] > 0 and action[1] == 0:       # left
            left += 1
        elif action[0] > 0 and action[1] > 0:      # left throttle
            left_throttle += 1
        elif action[0] < 0 and action[1] == 0:     # right
            right += 1
        elif action[0] > 0 and action[1] > 0:      # right throttle
            right_throttle += 1
        elif action[0] == 0 and action[1] > 0:     # throttle - accelerate
            acc += 1
        elif action[0] == 0 and action[2] > 0:     # brake
            brakes += 1
        # elif action[0] == 0 and action[1] == 0 and action[2] == 0:     # keep   #  why do we need "keep" folder?
        #     keep += 1
    summ = left+left_throttle+right+right_throttle+acc+brakes   # +keep
    print("====================================================")
    print("----------- Data pairs in total =", str(len(actions)), "------------")
    print("----------- Data pairs be used =", str(summ), "-------------")
    print("====================================================")
    print("Left = ", left)
    print("Left_throttle = ", left_brake)
    print("Right = ", right)
    print("Right_throttle = ", right_brake)
    print("Accelerate = ", acc)
    print("Break = ", brakes)
    # print("Keep = ", keep)
    print("====================================================")

def data_analyse_four(data_folder):
    observations, actions = load_imitations(data_folder)
    left, right, acc, brakes = 0,0,0,0
    for action in actions:
        if action[0] < 0:
            right += 1       # right
        elif action[0] > 0:
            left += 1        # left
        else:
            if action[1] > 0:
                acc += 1     # gas
            elif action[1] ==0 and action[2] > 0:
                brakes += 1  # brake
    summ = left+right+acc+brakes
    print("====================================================")
    print("----------- Data pairs in total =", str(len(actions)), "------------")
    print("----------- Data pairs be used =", str(summ), "-------------")
    print("====================================================")
    print("Left = ", left)
    print("Right = ", right)
    print("Accelerate = ", acc)
    print("Break = ", brakes)
    print("====================================================")

def data_analyse_nine(data_folder):
    observations, actions = load_imitations(data_folder)
    left, left_brake, left_brake_gas, \
    right, right_brake, right_brake_gas, \
    acc, brakes, keep = 0,0,0,0,0,0,0,0,0

    for action in actions:
        if action[0] < 0 and action[1] > 0:
            left_brake_gas+=1    # steering left and gas
        elif action[0] < 0 and action[1] ==0 and action[2] ==0:
            left+=1              # steering left
        elif action[0] < 0 and action[1] ==0 and action[2] > 0:
            left_brake+=1        # steering left and brake

        elif action[0] > 0 and action[1] > 0:
            right_brake_gas+=1    # steering right and gas
        elif action[0] > 0 and action[1] ==0 and action[2] ==0:
            right+=1              # steering right
        elif action[0] > 0 and action[1] ==0 and action[2] > 0:
            right_brake+=1        # steering right and brake

        elif action[0] == 0 and action[1] == 0 and action[2] == 0:
            keep+=1               # keep forward
        elif action[0] == 0 and action[2] > 0:
            brakes+=1             # brake
        elif action[0] == 0 and action[1] > 0 and action[2] == 0:
            acc+=1                # gas
    summ = left+left_brake+left_brake_gas+right+right_brake+right_brake_gas+acc+brakes+keep
    print("====================================================")
    print("----------- Data pairs in total =", str(len(actions)), "------------")
    print("----------- Data pairs be used =", str(summ), "-------------")
    print("====================================================")
    print("Left = ", left)
    print("Left_brake = ", left_brake)
    print("Left_brake_gas = ", left_brake_gas)
    print("Right = ", right)
    print("Right_brake_gas = ", right_brake_gas)
    print("Left_brake = ", left_brake)
    print("Accelerate = ", acc)
    print("Break = ", brakes)
    print("Keep = ", keep)
    print("====================================================")

# data_analyse_four('data/teacher')
# data_analyse_seven('data/teacher')
# data_analyse_nine('data/teacher')
