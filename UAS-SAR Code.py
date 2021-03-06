Working in fourier spaces
What functions do we need?
What should I do when the platform position data is not perfectly aligned with the pulses?
What will I do when there is a bias in the range-position and pulse time stamp?
import scipy.integrate as integrate
import scipy.special as special
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 8]   # make the default figure size bigger

# save pkl file data
pkl_file = open('2Points_data.pkl', 'rb')
platform_position, pulses, range_axis = pickle.load(pkl_file)
pkl_file.close()
pulses = pulses.tolist()

#how many range bins are in one meter???
total_distance = range_axis[0][1083] - range_axis[0][0]
bins_per_meter = len(range_axis[0]) / total_distance 
meters_per_bin = 1 / bins_per_meter

print(meters_per_bin)

#graphs signal intensity
def graph_signal(wave_index):
    x = np.arange(0, 1084)
    y = []
    for pul in pulses[wave_index]:
        y.append(abs(pul))
    plt.figure(1)
    plt.plot(x, y,'r-')
    plt.title('Signal Intensity')
    plt.xlabel('Range Bin')
    plt.ylabel('Magnitude');
    plt.show()

def range_delay(pulse_index, x, y, z = 0):
    
    ref_cord = [x, y, z]
    first_pt = platform_position[0]
    second_pt = platform_position[pulse_index]
    dist_between_scans = first_pt[0] - second_pt[0]
    
    range_1 = math.sqrt((first_pt[0] - ref_cord[0]) ** 2 + (first_pt[1] - ref_cord[1]) ** 2 + (first_pt[2] - ref_cord[2]) ** 2)
    #range_1_final = math.sqrt(range_1 ** 2 + first_pt[2] ** 2)
    
    range_2 = math.sqrt((second_pt[0] - ref_cord[0]) ** 2 + (second_pt[1] - ref_cord[1]) ** 2 + (second_pt[2] - ref_cord[2]) ** 2)
    #range_2_final = math.sqrt(range_2 ** 2 + second_pt[2] ** 2)
 
    return range_2 - range_1

def __range__(x, y, z = 0):
    ref_cord = [x, y, z]
    first_pt = platform_position[0]
    range_1 = math.sqrt((first_pt[0] - ref_cord[0]) ** 2 + (first_pt[1] - ref_cord[1]) ** 2 + (first_pt[2] - ref_cord[2]) ** 2)
    
    return range_1

def pixel_intensity(x, y, z = 0):
    
    #determines how many bins to shift based on the range delay
    bin_shift = []
    for ii in range(0, 100):
        bin_shift.append(int(round(range_delay(ii, x, y) * bins_per_meter)))
        
    #removes the necessary amount of bins to make the pulses aligned    
    for i in range(0, 100):
        for num in range(0, bin_shift[i]):
            if bin_shift[i] > 0:
                pulses[i].pop(0)
                pulses[i].append(0)
            else:
                pulses[i].pop(0)
                pulses[i].append(0)
    
    #finds range of the reference pulse to target pixel and figures out which bin is matched to that object
    bin_idx = int(round(__range__(x, y) * bins_per_meter))
    collapsed_pulses = (abs(np.sum(pulses, axis=0)))
    return collapsed_pulses[bin_idx]

image_pixels = np.zeros((50, 50))
row = 0
col = 0

for y in np.arange(0, 39.2, 0.4 * 2):
    print(y)
    if col != 0:
        row = row + 1
    col = 0
    for x in np.arange(-4, 3.92, 0.0792 * 2):
        image_pixels[row][col] = pixel_intensity(x, y)
        col = col + 1

print(np.ndim(image_pixels))
print(len(image_pixels))

plt.imshow(image_pixels)
plt.colorbar()
plt.show()

'''
np.argmin(np.abs(R-r))
returns index of smallest absolute value of R = range bins  array, r is values you are looking for
'''

