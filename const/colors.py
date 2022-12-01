import numpy as np
import os

np.random.seed(0)

def get_random_colors():

    if os.path.exists('random_colors.npy'):
        return np.load('random_colors.npy')
    else:
        nb_colors = 100000
        colors    = np.random.randint(0,255,(nb_colors,3))
        return colors
