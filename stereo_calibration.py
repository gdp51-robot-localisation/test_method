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
import numpy as np
import yaml
np.set_printoptions(threshold=np.inf)


if len( sys.argv ) < 2:
    print("Usage: laser line extraction Path/to/Image")
    sys.exit( -1 )


elif not os.path.exists( sys.argv[1] ):
    print("File not found")
    sys.exit( -1 )

yaml_folder_path = sys.argv[1]
yaml_file_name = sys.argv[2]

config = yaml.load(open(os.path.join(yaml_folder_path, yaml_file_name), 'r'))


folder_path_1 = config['Stereo_Calibration']['Camera1']['path']
file_name_1 = config['Stereo_Calibration']['Camera1']['filenamelist']
troubleshoot_file_name = config['Stereo_Calibration']['Troubleshooting']['sampleimage']
grid_size = config['Stereo_Calibration']['Calibrationgridsize']
grid_size = int(grid_size)
shape = (grid_size, grid_size)
distance_between_circles = config['Stereo_Calibration']['Gridspacing']
distance_between_circles = float(distance_between_circles)

min_blur = 0.0

show_found_grid = 0

file_name_format = file_name_1.split('.')


sub_path_1 = folder_path_1.split('/')        
sub_out_1=sub_path_1
outpath_1=sub_out_1[0]

for i in range(1,len(sub_path_1)):
    if sub_path_1[i]=='raw':
        sub_out_1[i] = 'processed'
#        proc_flag = 1
    else:
        sub_out_1[i] = sub_path_1[i]
        
    outpath_1 = outpath_1 +'/' + sub_out_1[i]
    # make the new directories after 'processed' if it doesnt already exist
#    if proc_flag == 1:        
    if os.path.isdir(outpath_1) == 0:
        try:
            os.mkdir(outpath_1)
        except Exception as e:
            print("Warning:",e)


params = cv2.SimpleBlobDetector_Params()
params.minArea = 10;
params.minDistBetweenBlobs = 5;
detector = cv2.SimpleBlobDetector_create(params)




# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp_1 = np.zeros((grid_size*grid_size,3), np.float32)
objp_1[:,:2] = np.mgrid[0:(grid_size):1,0:(grid_size):1].T.reshape(-1,2)*distance_between_circles

# Arrays to store object points and image points from all the images.
objpoints_1 = [] # 3d point in real world space
imgpoints_1 = [] # 2d points in image plane.

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

#Load image in greyscale
with open(os.path.join(folder_path_1, file_name_1), "r") as ifile:
    filelist = [line.rstrip() for line in ifile]
    for i in filelist:
        img = cv2.imread(os.path.join(folder_path_1, i), 0)
        (thresh_1, im_bw_1) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        im_bw_inverted_1 = cv2.bitwise_not(im_bw_1)
#        img_1_blur = cv2.Laplacian(im_bw_inverted_1, cv2.CV_64F).var()
#        print ("Processing image:", i, "Blur=", img_1_blur)
        [isFound_1, centers_1] = cv2.findCirclesGrid(im_bw_inverted_1, shape, flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING, blobDetector = detector)
        
        # If found, add object points, image points (after refining them)
        if isFound_1 == True:
            objpoints_1.append(objp_1)
#            cv2.cornerSubPix(im_bw_inverted_1,centers_1,(3,3),(-1,-1),subpix_criteria)
            imgpoints_1.append(centers_1)
            if show_found_grid == 1:
                img = cv2.drawChessboardCorners(im_bw_inverted_1, (7,7), centers_1 ,isFound_1)
                cv2.imshow('img_1_pre',img)
                cv2.waitKey(0)
        

print("unobj1:", len(objpoints_1))
print("unimg1:", len(imgpoints_1))

ret_1, mtx_1, dist_1, rvecs_1, tvecs_1 = cv2.calibrateCamera(objpoints_1, imgpoints_1, im_bw_inverted_1.shape[::-1],None,None)

print ("Matrix:", mtx_1)



img_1 = cv2.imread(os.path.join(folder_path_1, troubleshoot_file_name))
h,  w = img_1.shape[:2]

fov_x_1 = 2 * math.atan(math.radians(w/(2 * (mtx_1[0,0]))))
fov_y_1 = 2 * math.atan(math.radians(h/(2 * (mtx_1[1,1]))))


newcameramtx_1, roi_1=cv2.getOptimalNewCameraMatrix(mtx_1,dist_1,(w,h),1,(w,h))
#
# undistort
dst_1 = cv2.undistort(img_1, mtx_1, dist_1, None, newcameramtx_1)

# crop the image
x,y,w,h = roi_1
dst_1 = dst_1[y:y+h, x:x+w]
cv2.imwrite(os.path.join(outpath_1, troubleshoot_file_name), dst_1)


#Write image
troubleshoot_file_name_format = troubleshoot_file_name.split('.')
cv2.imwrite(os.path.join(outpath_1, troubleshoot_file_name_format[0] + "_original" + ".jpg"), img_orignal)
cv2.imwrite(os.path.join(outpath_1, troubleshoot_file_name_format[0] + "_undistorted" + ".jpg"), dst_1)

with open(os.path.join(outpath_1, 'camera_calibration.yaml'), 'w') as f:
    yaml.dump({'ret_1': ret_1,'cameraMatrix1': cameraMatrix1.tolist(),'newcameramtx_1': newcameramtx_1.tolist(),'mtx_1': mtx_1.tolist(), 'dist_1': dist_1.tolist(), 'fox_x': fov_x_1, 'fox_y': fov_y_1}, f)











