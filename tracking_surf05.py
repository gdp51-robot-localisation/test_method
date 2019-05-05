import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os, os.path
import math
import time
from datetime import datetime
startTime = datetime.now()

MIN_MATCH_COUNT = 10   # default=10

img1 = cv2.imread('Pattern3_small.jpg',0)          # queryImage

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
hessian_input = 2700
surf = cv2.xfeatures2d.SURF_create(hessianThreshold = hessian_input, extended = 1)
# upright = false

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("output_QT_1080.mov")

# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(img1,None)

pts_global = []
dst_global = []

position_raw = []
position = []
heading = []
reading_time = []
fps_array = []
# plt.axis([0, 1280, 0, 720])

mtx_1 = np.array([[7615.086684842451, 0.0, 934.6619126632753],[0.0, 7589.141593519471, 560.8448319169089],[0.0, 0.0, 1.0]])
dist_1 = np.array([3.3036628821574143, -284.6056262111876, -0.00990095676339995, -0.01422899829406913, -3.8589533787510892])

# # Robot test 24/04/19
bl = np.float32([283.0,1028.0])
br = np.float32([1596.3,1030.2])
tl = np.float32([286.5,46.7])
tr = np.float32([1595.3,44.3])

# # Second year Eurobot test 25/04/19
# bl = np.float32([177,1118])
# br = np.float32([1693,1124.12045])
# tl = np.float32([182.8,-48])
# tr = np.float32([1687,-65])

rect = (tl, tr, bl, br)
rect = np.array(rect)

# # compute the height and width of the new image
maxWidth=1200
maxHeight= round(0.75 * maxWidth)


# now that we have the dimensions of the new image, construct
# the set of destination points to obtain a "birds eye view",
# (i.e. top-down view) of the image, again specifying points
# in the top-left, top-right, bottom-right, and bottom-left
# order
dst = np.array([
[0, 0],
[maxWidth - 1, 0],
[0, maxHeight - 1],
[maxWidth - 1, maxHeight - 1]], dtype = "float32")

# compute the perspective transform matrix and then apply it
M_persp = cv2.getPerspectiveTransform(rect, dst)

while True:
    _, img2 = cap.read()
    img2 = cv2.undistort(img2, mtx_1, dist_1)
    img2 = cv2.warpPerspective(img2, M_persp, (maxWidth, maxHeight))

    # Start timer
    timer = cv2.getTickCount()

    # find the keypoints and descriptors with SIFT
    kp2, des2 = surf.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # store all the good matches as per Lowe's ratio test.
    good = []

    # ratio test as per Lowe's paper
    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < 0.7*n.distance:
                good.append(m)
        except ValueError:
            pass

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        A = np.array([dst[1,0,0],dst[1,0,1]])
        B = np.array([dst[1,0,0],0])
        C = np.array([dst[0,0,0],dst[0,0,1]])

        AB = np.subtract(B,A)
        AC = np.subtract(C,A)
        AB_length = np.linalg.norm(AB)
        AC_length = np.linalg.norm(AC)

        robot_angle = math.degrees(math.acos((np.dot(AB,AC))/(AB_length * AC_length)))

        if AC[0] < 0:
            robot_angle = 360 - robot_angle

        cX1 = float(((dst[2,0,0] - dst[0,0,0])/2.0) + dst[0,0,0])
        # print(cX1)
        cY1 = float(((dst[2,0,1] - dst[0,0,1])/2.0) + dst[0,0,1])
        # print(cY1)
        cX2 = float(((dst[3,0,0] - dst[1,0,0])/2.0) + dst[1,0,0])
        # print(cX2)
        cY2 = float(((dst[3,0,1] - dst[1,0,1])/2.0) + dst[1,0,1])
        # print(cY2)
        cX_raw = (cX1 + cX2)/2.0
        cY_raw = (cY1 + cY2)/2.0
        # print (cX)
        # print (cY)

        cX = 2000 * (cX_raw / maxWidth)
        cY = 1500 - (1500 * (cY_raw / maxHeight))

        position_raw.extend([cX_raw, cY_raw])
        position.extend([cX,cY])
        heading.append(robot_angle)
        time_delta = datetime.now() - startTime
        seconds_reading = time_delta.total_seconds()
        reading_time.append(seconds_reading)
        fps_array.append(fps)

        img2 = cv2.circle(img2, (dst[0,0,0], dst[0,0,1]), 10, (255,0,0))
        img2 = cv2.putText(img2, "Press ESC to finish test", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        img2 = cv2.putText(img2, "Heading: " + str(round(robot_angle)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        # img2 = cv2.putText(img2, "Position: " + str(int(position[(len(position) - 1)])), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        position_raw_formatted = np.array(position_raw)
        position_raw_formatted = np.int16(position_raw_formatted).reshape(-1,2)

        if len(position_raw_formatted) > 2:
            for i in range(2, (len(position_raw_formatted) - 1)):
                img2 = cv2.line(img2,tuple(position_raw_formatted[i-1]),tuple(position_raw_formatted[i]),(0,255,0),thickness = 3)

    else:
        # print ("Not enough matches are found - %d/%d") % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    # Display FPS on frame
    cv2.putText(img2, "FPS : " + str(round(fps,2)), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    scale_percent = 92 # percent of original size
    width_resized = int(img2.shape[1] * scale_percent / 100)
    height_resized = int(img2.shape[0] * scale_percent / 100)
    dim_resized = (width_resized, height_resized)
    # resize image
    img2_resized = cv2.resize(img2, dim_resized, interpolation = cv2.INTER_AREA)
    # cv2.imshow("Frame", img2)
    cv2.imshow("Frame", img2_resized)
    # cv2.imshow("Frame", img3)

    key = cv2.waitKey(1)
    if key == 27:   #ESC
        break
        
    
cap.release()
cv2.destroyAllWindows()

position = np.array(position)
position = position.reshape(-1,2)
heading = np.array(heading)
heading = heading.reshape(-1,1)
reading_time = np.array(reading_time)
reading_time = reading_time.reshape(-1,1)
fps_array = np.array(fps_array)
fps_array = fps_array.reshape(-1,1)

x, y = position.T

plt.plot(x, y)
plt.xlim(0, 2000)
plt.ylim(0, 1500)
plt.xlabel('Width (mm)')
plt.ylabel('Height (mm)')
plt.title('Position')
plt.grid()
# plt.savefig("test.png")
plt.show()

print("Time taken:", datetime.now() - startTime)
print("Start position:", position[0])
print("Finish position:", position[(len(position) - 1)])
# print("Mean position:", np.mean(position,0))

timestr = time.strftime("%Y%m%d-%H%M%S")
position_plus_heading = np.concatenate((reading_time, position, heading, fps_array), axis=1)

np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "pts_" + timestr + ".csv"), position_plus_heading, delimiter=",")



