"""
07/02/2019
Michael Leat

Program to perform camera calibration and store parameters for later use.

Inputs:
1. Calibration images for camera 1 - path and file name list in .txt
2. File name for a troubleshooting output image to check calibration.
3. Calibration grid size. For a 7 x 7 grid, input is 7
4. Calibration grid spacing. The distance from the center to center of each 
circle on the calibration grid. Units are meters.

Running:
The program is run in terminal in the following way:
    
python stereo_calibration.py configuration_filepath configuration_filename.yaml 

e.g. 

python stereo_calibration.py /Volumes/SCALAR/git configuration.yaml


Output:
The function returns a .yaml file called camera_calibration.yaml containing 
the intrinsic camera parameters which can be stored for use in 
other programs.



"""


import cv2
import sys
import os, os.path
import time





folder_path = sys.argv[1]
no_images = sys.argv[2]

no_images = int(no_images)

if not os.path.exists(folder_path):
    print ("Creating directory")
    os.makedirs(folder_path)

cap = cv2.VideoCapture(0)

for i in range(1,no_images+1):
    print ("Saving image", i, "of", no_images)
    time.sleep(7)
    _, img1 = cap.read()
    i = str(i)
    cv2.imwrite(os.path.join(folder_path, ("image" + i + ".jpg")), img1)
    os.system( "say Next" )


