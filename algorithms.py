import pickle 
import matplotlib.pyplot as plt
import numpy as np
import pandas as panda
import sys
import argparse
import os
from scipy.misc import imsave
SPEED_OF_LIGHT = 299792458


def align_motion_capture(platform_position, motion_capture_time):
    
    # once the radar moves more than 0.02 m in 20 frames, the radar is assumed to have started recording
    for ii in range(0, len(motion_capture_time) - 20):
        if abs(platform_position[ii, 0] - platform_position[ii + 20, 0]) > 0.02 or \
        abs(platform_position[ii, 1] - platform_position[ii + 20, 1]) > 0.02 or \
        abs(platform_position[ii, 2] - platform_position[ii + 20, 2]) > 0.02:
            return ii
    
    return 'NaN'


def find_moving_radar(scan_data):
    
    # index where radar starts/stops to move (already recording data)
    start_index = 0 
    stop_index = 0
    
    # computes absolute difference between 'interval' amount of pulses
    # finds where radar starts to move
    interval = 5
    for ii in range(0, len(scan_data) - interval, interval):
        z = sum(abs((scan_data[ii] - scan_data[ii + interval])))
        if z > 200000:
            start_index = ii
            break

    # finds where radar stops moving
    for jj in range(len(scan_data) - 1, 0, -interval):
        v = sum(abs((scan_data[jj] - scan_data[jj - interval])))
        if v > 200000:
            stop_index = jj
            break
    
    return start_index, stop_index  


# where the motion capture system determines the radar has stopped moving (index) 
def motion_capture_stop_index(duration_of_radar_movement, motion_capture_time, m = 'align motion capture'):

    for kk in range(m, len(motion_capture_time)):
        if motion_capture_time[kk] - motion_capture_time[m] >= duration_of_radar_movement:
            return kk


def re_adjust_platform_position(duration_of_radar_movement, platform_position, motion_capture_time):
    
    p = []
    desired_length = 1887  # need to make this modular
    current_length = len(platform_position[align_motion_capture(platform_position, motion_capture_time):motion_capture_stop_index(duration_of_radar_movement, motion_capture_time, align_motion_capture(platform_position, motion_capture_time))])
    interval = current_length / desired_length
    for hh in range(0, desired_length):
        r = int(round(hh * interval))
        y = platform_position[r]
        p.append(y)
    return p


def fourier_approach(pulses, range_axis, platform_pos, x_vec, y_vec, center_freq):
    """
    Backprojection using shifts implemented through linear phase ramps.
    """
    # Determine dimensions of data
    (num_pulses, num_range_bins) = pulses.shape
    num_x_pos = len(x_vec)
    num_y_pos = len(y_vec)
    
    # Compute the fast-time or range-bin times
    fast_time = np.transpose(range_axis / SPEED_OF_LIGHT)
    delta_fast_time = fast_time[1] - fast_time[0]
    
    # Compute the unwrapped angular frequency
    ang_freq = np.transpose(2 * np.pi * 
                            np.arange(-num_range_bins / 2, num_range_bins / 2) / 
                            (delta_fast_time * num_range_bins))
    
    # X-Y locations of image grid
    x_grid, y_grid = np.meshgrid(x_vec, y_vec)
    
    # Initialize SAR image
    complex_image = np.zeros_like(x_grid, dtype=np.complex)
    
    # Iterate over each X-position in image grid and focus all the pixels 
    # across the Y-span of the image grid, i.e., a column
    for ii in range(0, num_x_pos):
        print('%d of %d' % (ii, num_x_pos))
        
        # Initialize current column's sum of aligned pulses
        sum_aligned_pulses = np.zeros(num_y_pos, dtype=np.complex)
        
        # Iterate over each pulse
        for jj in range(0, num_pulses):
            
            # Calculate the 2-way time delay to each point in the current 
            # column of the image grid
            two_way_time = 2 * np.sqrt(
                    (x_grid[:, ii] - platform_pos[jj, 0])**2 + 
                    (y_grid[:, ii] - platform_pos[jj, 1])**2 +
                    platform_pos[jj, 2]**2) / SPEED_OF_LIGHT
                    
            # Demodulate the current pulse
            demod_pulse = (np.transpose(np.atleast_2d(pulses[jj, :])) * 
                           np.exp(-1j * 2 * np.pi * center_freq * 
                                  (fast_time - two_way_time)))
            
            # Align the current pulses contribution to current column
            demod_pulse_freq = np.fft.fftshift(np.fft.fft(demod_pulse, axis=0),
                                               axes=0)
            phase_shift = np.exp(1j * np.outer(ang_freq, two_way_time))
            demod_pulse_freq_aligned = phase_shift * demod_pulse_freq
            pulse_aligned = np.fft.ifft(
                    np.fft.ifftshift(demod_pulse_freq_aligned, 0), axis=0)
            
            # Update current column's sum of aligned pulses
            sum_aligned_pulses += np.transpose(pulse_aligned[0])
            
        # Update complex image with latest column's result
        complex_image[:, ii] = sum_aligned_pulses
        
    return complex_image


def main(args):
    """
    Top level methods
    """
    # Parse input arguments
    parsed_args = parse_args(args)

  
    
    # download the data from the radar
    f = open('railTest1.pkl', 'rb')
    radar_data = pickle.load(f)
    f.close()

    # unpack radar pickle file
    scan_data, time_stamp, packet_ind, packet_pulse_ind, range_bins = radar_data.values()

    # download the data from the motion capture system
    data_file = panda.read_csv('UASSAR4_rail_1.csv', skiprows = 6)
    motion_capture_data = data_file.values
    
    platform_position = motion_capture_data[:, [6, 7, 8]]
    
    # saves time stamps from motion capture system in a single array
    motion_capture_time = motion_capture_data[:,1]

    a, b = find_moving_radar(scan_data)
    duration_of_radar_movement = (time_stamp[b] - time_stamp[a]) / 1000


    
    # saves all of the platform positions as : by 3 array from the motion capture data

    
    # find how long radar was moving for and convert to seconds
    
    
    #need to tranpspose platform position array
    pulses = scan_data[a:b] # len = 1887
    #new_time_stamps = time_stamp[a:b] - time_stamp[a] # len = 1887
    platform_pos =  re_adjust_platform_position(duration_of_radar_movement, platform_position, motion_capture_time)
    range_axis = range_bins
    
    
    # Determine X-Y coordinates of image pixels
    x_vec = np.arange(parsed_args.x_bounds[0], parsed_args.x_bounds[1], 
                      parsed_args.pixel_res)
    y_vec = np.arange(parsed_args.y_bounds[0], parsed_args.y_bounds[1], 
                      parsed_args.pixel_res)
    
    # Form SAR image
    if parsed_args.method == 'shift':
        complex_image = shift_approach(
                pulses, range_axis, platform_pos, x_vec, y_vec)
    elif parsed_args.method == 'interp':
        complex_image = interp_approach(
                pulses, range_axis, platform_pos, x_vec, y_vec)
    elif parsed_args.method == 'fourier':
        complex_image = fourier_approach(
                pulses, range_axis, platform_pos, x_vec, y_vec, 
                parsed_args.center_freq)
    else:
        raise ValueError('Unknown method %s specified' % parsed_args.method)    
        
    # Convert to magnitude image for visualization
    image = np.abs(complex_image)
        
    # Show SAR image
    if not parsed_args.no_visualize:
        image_extent = (x_vec[0], x_vec[-1], y_vec[0], y_vec[-1])
        plt.figure()
        plt.subplot(121)
        plt.imshow(image, origin='lower', extent=image_extent)
        plt.title('Linear Scale')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(20 * np.log10(image), origin='lower', extent=image_extent)
        plt.title('Logarithmic Scale')
        plt.colorbar()
        plt.show()
        
    # Save image
    imsave(parsed_args.output, image)


def parse_args(args):
    """
    Input argument parser.
    """
    parser = argparse.ArgumentParser(
            description=('SAR image formation via backprojection'))
    parser.add_argument('input', nargs='?', type=str,
                        help='Pickle containing data')
    parser.add_argument('x_bounds', nargs=2, type=float, 
                        help=('Minimum and maximum bounds of the X coordinates'
                              ' of the image (m)'))
    parser.add_argument('y_bounds', nargs=2, type=float, 
                        help=('Minimum and maximum bounds of the Y coordinates'
                              ' of the image (m)'))
    parser.add_argument('pixel_res', type=float, help='Pixel resolution (m)')
    parser.add_argument('-o', '--output', nargs='?', const=None, default=None, 
                        type=str, help='File to store SAR image to')
    parser.add_argument('-m', '--method', nargs='?', type=str,
                        choices=('shift', 'interp', 'fourier'),
                        default='fourier', const='fourier', 
                        help='Backprojection method to use')
    parser.add_argument('-fc', '--center_freq', type=float, 
                        help=('Center frequency (Hz) of radar; must be '
                              'specified if using fourier method'))
    parser.add_argument('-nv', '--no_visualize', action='store_true', 
                        help='Do not show SAR image')
    parsed_args = parser.parse_args(args)
    
    # Do some additional checks
    if parsed_args.output is None:
        root, ext = os.path.splitext(parsed_args.input)
        parsed_args.output = '%s.png' % root
    
    return parsed_args


if __name__ == "__main__":
    """
    Standard Python alias for command line execution.
    """
    main(sys.argv[1:])
