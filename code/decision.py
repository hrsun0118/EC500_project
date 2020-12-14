import numpy as np

# added later
import os
import torch

directory = "C:\\D drive\\Fall 2020\\EC500\\project\\EC500_project\\code\\"
trained_network_file = os.path.join(directory, 'data\\train.t7')

# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):
    # update action based on current observation
    infer_action = torch.load(trained_network_file, map_location='cuda') # original 'cpu'
    infer_action.eval()
    device = torch.device('cuda')   # ofiginal 'cpu'
    infer_action = infer_action.to(device)
    observation = Rover.img
    action_scores = infer_action(torch.Tensor(
        np.ascontiguousarray(observation[None])).to(device))
    Rover.steer, Rover.throttle, Rover.brake = infer_action.scores_to_action(action_scores)
    # print("Rover.steer=",Rover.steer, "Rover.throttle=", Rover.throttle, "Rover.brake=", Rover.brake)
    return Rover
