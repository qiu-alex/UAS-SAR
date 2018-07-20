import pickle 
import matplotlib.pyplot as plt
import numpy as np
import math

f = open("2Points_data.pkl", "rb")
platform_position, pulses, range_axis = pickle.load(f)
f.close()

def __range__(pulse_index, x, y, z):
    first_pt = platform_position[pulse_index] # this is the plane position at that pulse
    return math.sqrt((first_pt[0] - x) ** 2 + (first_pt[1] - y) ** 2 + (first_pt[2] - z) ** 2)

def num_range_bin(range):
    return round(range/0.0184615) #interpolation, rounding errors to smooth out image
# meters per bin, one way range = bins 

list_intensities = []
for y in np.arange(-3, 3, 0.01):
    for x in np.arange(-3, 3, 0.01):
        intensity_final = 0
        for ii in range(0, 100):
            intensity_final+= pulses[ii][num_range_bin(__range__(ii, x, y, 0))]
        list_intensities.append(abs(intensity_final))
        print(x, y)

sar = np.reshape(list_intensities, (600, 600))
plt.imshow(np.flip(sar, 0))
plt.colorbar()
plt.show()
