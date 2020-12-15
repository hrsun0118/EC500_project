import os
import numpy as np

# added
import pandas as pd
from PIL import Image

def get_vision_img(image):
    vision_image = np.zeros((160, 320, 3), dtype=np.float)
    worldmap = np.zeros((200, 200, 3), dtype=np.float)
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5
    # Set a bottom offset to account for the fact that the bottom of the image
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      ])
    warped, mask = perspect_transform(image, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped)
    obs_map = np.absolute(np.float32(threshed) - 1) * mask
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    vision_image[:,:,2] = threshed * 255
    vision_image[:,:,0] = obs_map * 255
    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)
    # 6) Convert rover-centric pixel values to world coordinates
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1],
                                    Rover.yaw, world_size, scale)
    obsxpix, obsypix = rover_coords(obs_map)
    obs_x_world, obs_y_world = pix_to_world(obsxpix, obsypix, Rover.pos[0], Rover.pos[1],
                                            Rover.yaw, world_size, scale)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[y_world, x_world, 2] += 10
    Rover.worldmap[obs_y_world, obs_x_world, 2] += 1
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    dist, angles = to_polar_coords(xpix, ypix)
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    Rover.nav_angles = angles

    # See if we can find some rocks
    rock_map = find_rocks(warped, levels = (110, 110, 50))
    if rock_map.any():
        rock_x, rock_y = rover_coords(rock_map)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, Rover.pos[0],
                                                    Rover.pos[1], Rover.yaw, world_size, scale)
        rock_dist, rock_ang = to_polar_coords(rock_x, rock_y)
        rock_idx = np.argmin(rock_dist)
        rock_xcen = rock_x_world[rock_idx]
        rock_ycen = rock_y_world[rock_idx]

        Rover.worldmap[rock_ycen, rock_xcen, 1] = 255
        Rover.vision_image[:, :, 1] = rock_map * 255
    else:
        Rover.vision_image[:, :, 1] = 0

    return Rover

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
    actions = np.asarray(csv_arr[:, 1:4], dtype=np.float16)
    #print("actions[0]: ", actions[0], "; action[1]:", actions[1])

    #print(actions.dtype)

    # get observations
    obs_files = os.listdir(data_folder + '\\IMG\\')
    observations = [0]*int(len(obs_files))  # create list
    index = 0
    for filename in obs_files:  # loop through all files
        open_file_name = os.path.join(os.path.join(data_folder + '/IMG/'), filename)
        # originally: observations[index] = np.asarray(Image.open(open_file_name))
        # now:
        front_cam_img = np.asarray(Image.open(open_file_name))
        observations[index] = get_vision_img(front_cam_img)
        index += 1
    observations = np.asarray(observations)
    #print("observations[0]: ", observations[0])
    # print("observations[0] shape: ", observations[0].shape)

    return observations, actions

# following code is for testing purpose only, need to be commented out later
# data_folder = '/Users/hairuosun/Library/Mobile Documents/com~apple~CloudDocs/BU/Fall 2020/Courses/EC 500 A2/Project/Github Simulation/EC500_project/data/'
# load_imitations(data_folder)
