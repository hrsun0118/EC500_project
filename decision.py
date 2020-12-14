import numpy as np

# added later
import os
import torch

directory = "C:\\D drive\\Fall 2020\\EC500\\project\\EC500_project\\"
trained_network_file = os.path.join(directory, 'data\\train.t7')

def evaluate():
    infer_action = torch.load(trained_network_file, map_location='cpu')
    infer_action.eval()
    # env = gym.make('CarRacing-v0')
    # you can set it to torch.device('cuda') in case you have a gpu
    device = torch.device('cpu')
    infer_action = infer_action.to(device)


    #for episode in range(5):
        #observation = env.reset()

        #reward_per_episode = 0
        #for t in range(500):
            #env.render()
            action_scores = infer_action(torch.Tensor(
                np.ascontiguousarray(observation[None])).to(device))

            steer, gas, brake = infer_action.scores_to_action(action_scores)
            #observation, reward, done, info = env.step([steer, gas, brake])
            #reward_per_episode += reward

        #print('episode %d \t reward %f' % (episode, reward_per_episode))




# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):
    # update action based on current observation
    infer_action = torch.load(trained_network_file, map_location='cpu')
    infer_action.eval()
    device = torch.device('cuda')
    infer_action = infer_action.to(device)
    action_scores = infer_action(torch.Tensor(
        np.ascontiguousarray(observation[None])).to(device))
    Rover.steer, Rover.throttle, Rover.brake = infer_action.scores_to_action(action_scores)
    return Rover
