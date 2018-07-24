# -*- coding: utf-8 -*-

import pickle 
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from bisect import bisect_left

def __range__(platform_position, pulse_index, x, y, z):
    first_pt = platform_position[pulse_index] # this is the plane position at that pulse
    return math.sqrt((first_pt[0] - x) ** 2 + (first_pt[1] - y) ** 2 + (first_pt[2] - z) ** 2)

def num_range_bin(range):
    return range/0.0184615 # interpolation, rounding errors to smooth out image
# meters per bin, one way range = bins

def takeClosest(myList, myNumber): # finds the element in the list closest to the element given
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    return before, after

def platform_position_function(pulse_time_stamp, platform_time_stamp, platform_position): # lists associated with the values written
    platform_position_pulse2  = []
    for i in range(0, len(pulse_time_stamp)):
        # linear interpolation to find the point values at each pulse time stamp
        (a, b) = takeClosest(platform_time_stamp, pulse_time_stamp[i])
        c = pulse_time_stamp[i]
        r = platform_time_stamp.index(a)
        start = platform_position(r)
        end = platform_position(r+1)
        proportion = (c - a) / (b - a)
        
        positionx = ((end[0] - start[0]) * proportion) + start[0]
        positiony = ((end[1] - start[1]) * proportion) + start[1]
        positionz = ((end[2] - start[2]) * proportion) + start[2]

        platform_position_pulse2.append((positionx, positiony, positionz))
    return platform_position_pulse2

def sar_imaging(res, x, y, z):
    resx = float(res[0])
    resy = float(res[1])
    startx = int(x[0])
    endx = int(x[1])
    starty = int(y[0])
    endy = int(y[1])
    zvalue = int(z)
    f = open("mandrill_no_aliasing_data.pkl", "rb")
    data = pickle.load(f)
    platform_position = data[0]
    pulses = data[1]
    range_axis = data[2]
    f.close()

    platform_position_pulse = platform_position_function(pulse_time_stamp, platform_time_stamp, platform_position) # add the list values required for input
    list_intensities = []
    for y in np.arange(starty, endy, resy):
        for x in np.arange(startx, endx, resx):
            intensity_final = 0
            for ii in range(0, 100):
                # linear interpolation
                range_bin = num_range_bin(__range__(platform_position_pulse[ii], ii, x, y, zvalue))
                range_bin_floor = math.floor(num_range_bin(__range__(platform_position_pulse[ii], ii, x, y, zvalue)))
                range_bin_ceil = math.ceil(num_range_bin(__range__(platform_position_pulse[ii], ii, x, y, zvalue)))
                proportion = (range_bin - range_bin_floor) / (range_bin_ceil - range_bin_floor)
                intensity = ((pulses[ii][range_bin_ceil] - pulses[ii][range_bin_floor]) * proportion) + pulses[ii][range_bin_floor]
                intensity_final += intensity
            list_intensities.append(abs(intensity_final))
            print(x, y, z)

    sar = np.reshape(list_intensities, (int((endx-startx)/resx), int((endy-starty)/resy)))
    plt.imshow(np.flip(sar, 0), cmap=plt.get_cmap('gray'))
    plt.colorbar()
    plt.show()

if __name__ == "__main__":  
    sar_imaging((sys.argv[1], sys.argv[2]),  (sys.argv[3], sys.argv[4]), (sys.argv[5], sys.argv[6]), sys.argv[7])
