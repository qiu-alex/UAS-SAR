# -*- coding: utf-8 -*-

import pickle 
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import pandas as pd

def __range__(platform_position, pulse_index, x, y, z):
    first_pt = platform_position[pulse_index] # this is the plane position at that pulse
    return math.sqrt((first_pt[0] - x) ** 2 + (first_pt[1] - y) ** 2 + (first_pt[2] - z) ** 2)

def num_range_bin(range):
    return (range - 1.99991549)/(0.00914894 / 2) #interpolation, rounding errors to smooth out image, mkae modular
    # meters per bin, one way range = bins 

#def takeClosest(myList, myNumber): # finds the element in the list closest to the element given
#    pos = bisect_left(myList, myNumber)
#    if pos == 0:
#        return myList[0]
#    if pos == len(myList):
#        return myList[-1]
#    before = myList[pos - 1]
#    after = myList[pos]
#    return before, after

def takeClosest(time, value): #returns the index of where this value belongs
    for i in range(len(time)):
        if(value > time[i]):
            return i
    return 0

def get_platform_position(motionCapture, mcTime, timeStamp):
    #timeStamp is for the radar pulses
    #mcTime is for the motion capture times, interval is 6 ms.
    motionCapture = np.reshape(motionCapture, (int(len(motionCapture)/3), 3), order = 'F')
    platform_position = np.empty((len(timeStamp), 3))
    start_row_index = 1663
    end_row_index = 4892
    motionCaptureCopy = np.empty((end_row_index - start_row_index, 3))
    for i in range(start_row_index, end_row_index):
        motionCaptureCopy[i - start_row_index] = motionCapture[i] #fix all this indexing stuff
    motionCapture = motionCaptureCopy #remember in motion capture data the z is normal x, x is normal y, and y is normal z
    mcTimeCopy = np.empty((end_row_index - start_row_index))
    for i in range(start_row_index, end_row_index):
        mcTimeCopy[i - start_row_index] = mcTime[i]
    mcTime = mcTimeCopy
    mcTime -= mcTime[0]
    timeStamp = timeStamp / 1000.0
    for i in range(len(timeStamp)):
        mc = motionCapture[takeClosest(mcTime, timeStamp[i])]
        print(mc)
        x = mc[2]
        y = mc[0]
        z = mc[1]
        platform_position[i][0] = x
        platform_position[i][1] = y
        platform_position[i][2] = z
    return platform_position
    
    
    
    
def sar_imaging(res, x, y):
    resx = float(res[0])
    resy = float(res[1])
    startx = int(x[0])
    endx = int(x[1])
    starty = int(y[0])
    endy = int(y[1])
    
    #original
    f = open("mandrill_no_aliasing_data.pkl", "rb")
    data = pickle.load(f)
    platform_position = data[0]
   # pulses = data[1]
   # range_axis = data[2]
    f.close()
    
    #with motion capture
    f = open("railTest1.pkl", "rb")
    data = pickle.load(f)
    f.close()
    pulses = np.abs(np.asarray(data["scan_data"]))
    time_stamp = np.asarray(data["time_stamp"])
    # packet_ind = np.asarray(data["packet_ind"])
    # packet_pulse_ind = np.asarray(data["packet_pulse_ind"])
    range_axis = np.asarray(data["range_bins"])
    
    
    df = pd.read_csv('UASSAR4_rail_1.csv', skiprows=3)
    array = df.values
    mcTime = array[...,1]
    
    #finalList = np.zeros((6,))
    #count = 0
    #for i in range(len(array[0])):
    #    if(array[0][i] == 'UASSAR4' and count < 7):
    #        finalList[count] = i
    #        count+=1
    #finalList

    motionCapture = np.append(array[...,6], np.append(array[..., 7], array[..., 8]))
    time_stamp -= time_stamp[0]
    platform_position = get_platform_position(motionCapture, mcTime, time_stamp) #make sure to set up x, y, and z correspondingly
    print(platform_position)
    return
    list_intensities = []
    for y in np.arange(starty, endy, resy):
        for x in np.arange(startx, endx, resx):
            intensity_final = 0
            for ii in range(0, len(pulses)): # insert num range bins
                # linear interpolation
                range_bin = num_range_bin(__range__(platform_position, ii, x, y, 0))#make sure to include the z of the objects too
                range_bin_floor = math.floor(num_range_bin(__range__(platform_position, ii, x, y, 0)))
                range_bin_ceil = math.ceil(num_range_bin(__range__(platform_position, ii, x, y, 0)))
                proportion = (range_bin - range_bin_floor) / (range_bin_ceil - range_bin_floor)
                intensity = ((pulses[ii][range_bin_ceil] - pulses[ii][range_bin_floor]) * proportion) + pulses[ii][range_bin_floor]
                intensity_final += intensity
            list_intensities.append(abs(intensity_final))
            print(x, y)

    sar = np.reshape(list_intensities, (int((endx-startx)/resx), int((endy-starty)/resy)))
    #plt.imshow(np.flip(sar, 0), cmap=plt.get_cmap('gray'))
    plt.imshow(np.flip(sar, 0))
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    sar_imaging((sys.argv[1], sys.argv[2]),  (sys.argv[3], sys.argv[4]), (sys.argv[5], sys.argv[6]))
    