import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):
    # update action based on current observation
    Rover.steer, Rover.throttle, Rover.brake = ()

    return Rover
