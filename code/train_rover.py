import os
import sys
import numpy as np
import torch

from training import train
from imitations import record_imitations

directory = ""  ######## change that! ########
trained_network_file = os.path.join(directory, 'data/train.t7')
imitations_folder = os.path.join(directory, 'data/teacher')
