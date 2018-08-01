# -*- coding: utf-8 -*-

import pickle 
import matplotlib.pyplot as plt
import numpy as np
import math
#import sys
#import pandas as pd



def __range__(platform_position, pulse_index, x, y, z):
    first_pt = platform_position[pulse_index] # this is the plane position at that pulse
    return math.sqrt((first_pt[0] - x) ** 2 + (first_pt[1] - y) ** 2 + (first_pt[2] - z) ** 2)

def num_range_bin(range):
    return range/0.0184615 #interpolation, rounding errors to smooth out image
# meters per bin, one way range = bins 
def sar_imaging(res, x, y):
    resx = float(res[0])
    resy = float(res[1])
    startx = int(x[0])
    endx = int(x[1])
    starty = int(y[0])
    endy = int(y[1])
    #f = open("mandrill_no_aliasing_data.pkl", "rb")
    f = open('mandrill_no_aliasing_data.pkl','rb')
    data=pickle.load(f)
    f.close()
    platform_position = data[0]
    pulses = data[1]
    range_axis = data[2]
    f.close()

    list_intensities = []
    for y in np.arange(starty, endy, resy):
        for x in np.arange(startx, endx, resx):
            intensity_final = 0
            for ii in range(0, 100):
                # linear interpolation
                range_bin = num_range_bin(__range__(platform_position, ii, x, y, 0))
                range_bin_floor = math.floor(num_range_bin(__range__(platform_position, ii, x, y, 0)))
                range_bin_ceil = math.ceil(num_range_bin(__range__(platform_position, ii, x, y, 0)))
                proportion = (range_bin - range_bin_floor) / (range_bin_ceil - range_bin_floor)
                intensity = ((pulses[ii][range_bin_ceil] - pulses[ii][range_bin_floor]) * proportion) + pulses[ii][range_bin_floor]
                intensity_final += intensity
            list_intensities.append(abs(intensity_final))
            print(x, y)

    sar = np.reshape(list_intensities, (int((endx-startx)/resx), int((endy-starty)/resy)))
    sar = 45000 - sar
    plt.imshow(np.flip(sar, 0), cmap=plt.get_cmap('gray'))
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    sar_imaging((0.06, 0.06),  (-3, 3), (-3, 3))
    
