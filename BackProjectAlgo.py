# -*- coding: utf-8 -*-

import numpy as np

# platPosX, platPosY, platPosZ = platform position at a pulse index
# pulseNumber = the index of the pulse
# rangeBin = the range bin being called repeatedly, iterated

# these are inputs in the final equation
# final = the final value that will be returned
# iterate = the value used to calculate sum
# rangeBinOrig = range bin original, the range bin of the point being measured, at the first index
# pixelPosX = the pixel positions passed to the main function
# pixelPosY
# pixelPosZ 

# Goes into this function, but is fixed for all iterations: pixelPosX, pixelPosY, pixelPosZ
def rangeDelay(platPosX, platPosY, platPosZ):
   # calculates the range delay for a row using platPos and RangeBinOrig
   # previous formula  return(np.subtract(np.sqrt(np.add(np.square(platPos), np.square(RangeBinOrig))), RangeBinOrig))
   A = np.sqrt(np.add(np.square(np.subtract(platPosX, pixelPosX)), np.square(np.subtract(platPosY, pixelPosY)), np.square(np.subtract(platPosZ, pixelPosZ))))
   return(A)

def intensity(pulseNumber, rangeBin):
   # has to read the intensity with pulse number and the range bin
   return("null")

def platform(pulseNumber):
   # has to read the platform position with the pulse number
   return("null") # has to return three values, beacuse range delay needs x,y,z of plat position  

def pixelBrightness(rangeBinOrig, pixelPosX, pixelPosY, pixelPosZ):
   # calculates the brigtness of a pixel
   for pulseNumber in 0: # substitute 0 with number of rows
       pulseNumber = pulseNumber + 1
       iterate = intensity(pulseNumber, rangeBinOrig + rangeDelay(platform(pulseNumber)))
       # might be rangeBinOrig - rangeDelay(platform(pulseNumber), have to test if plus or minus
       final = final + iterate
   return final