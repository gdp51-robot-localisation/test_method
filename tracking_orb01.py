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
# img1 = cv2.imread('Pattern_robot.png',0)          # queryImage
# img2 = cv2.imread('Pattern4.jpg',0) # trainImage

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
# surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 4000)
orb = cv2.ORB_create()

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("output_H264_30.mov")

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)

pts_global = []
dst_global = []

position = []
heading = []
reading_time = []
fps_array = []
# plt.axis([0, 1280, 0, 720])

tbl_lower_horiz = 344 #343
tbl_upper_horiz = 1533 #1539
tbl_lower_vert = 95 #110
tbl_upper_vert = 987 #1008

mtx_1 = np.array([[7615.086684842451, 0.0, 934.6619126632753],[0.0, 7589.141593519471, 560.8448319169089],[0.0, 0.0, 1.0]])
dist_1 = np.array([3.3036628821574143, -284.6056262111876, -0.00990095676339995, -0.01422899829406913, -3.8589533787510892])

# cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Frame", 600,350)

while True:
    _, img2 = cap.read()
    img2 = cv2.undistort(img2, mtx_1, dist_1)

    # Start timer
    timer = cv2.getTickCount()

    # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12, 6
                   key_size = 12,     # 20, 12
                   multi_probe_level = 1) #2, 1
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # print (matches)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # store all the good matches as per Lowe's ratio test.
    good = []
    
    # for i in matches:
    #     if len(matches[i]) > 1:
    #         for m, n in matches:
    #             if m.distance < 0.7*n.distance:
    #                 good.append(m)
    #     if len(matches[i]) < 2:
    #         continue

    # Need to draw only good matches, so create a mask
    # matchesMask = [[0,0] for i in range(len(matches))]

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

        # print (dst)

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

        # print (robot_angle)
        cX = float(((dst[2,0,0] - dst[0,0,0])/2.0) + dst[0,0,0])
        cY = float(((dst[2,0,1] - dst[0,0,1])/2.0) + dst[0,0,1])
        # print (cX)
        # print (cY)

        # cX = [a - tbl_lower_horiz for a in x]
        # cX = [(2 * (a / (tbl_upper_horiz - tbl_lower_horiz))) for a in x]
        # cY = [a - tbl_lower_vert for a in y]
        # cY = [(1.5 * (a / (tbl_upper_vert - tbl_lower_vert))) for a in y]

        cX = 2000 * ((cX - tbl_lower_horiz) / (tbl_upper_horiz - tbl_lower_horiz))
        cY = 1500 - (1500 * ((cY - tbl_lower_vert) / (tbl_upper_vert - tbl_lower_vert)))

        position.extend([cX,cY])
        heading.append(robot_angle)
        time_delta = datetime.now() - startTime
        seconds_reading = time_delta.total_seconds()
        reading_time.append(seconds_reading)
        fps_array.append(fps)

        # sf = 0.04
        # scale_factor = sf/AC_length
        
        # AC_scaled = np.multiply(AC, scale_factor)
        # print(AC_scaled)

        # arrow = mpatches.FancyArrowPatch((cX,cY),(AC_scaled[0],AC_scaled[1]))
        # plt.figure()

        # plt.scatter(cX, cY)
        # plt.xlim(0, 2)
        # plt.ylim(1.5, 0)
        # plt.xlabel('Width (m)')
        # plt.ylabel('Height (m)')
        # plt.title('Position')
        # plt.arrow(cX,cY,AC_scaled[0],AC_scaled[1],head_width=(sf/2), head_length=(sf/2), fc='k', ec='k')
        # plt.pause(0.1)

        img2 = cv2.circle(img2, (dst[0,0,0], dst[0,0,1]), 10, (255,0,0))
        img2 = cv2.putText(img2, "Press ESC to finish test", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        img2 = cv2.putText(img2, "Heading: " + str(int(robot_angle)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        img2 = cv2.line(img2,(tbl_lower_horiz,tbl_lower_vert),(tbl_upper_horiz,tbl_lower_vert),(255,0,0),1)
        img2 = cv2.line(img2,(tbl_lower_horiz,tbl_upper_vert),(tbl_upper_horiz,tbl_upper_vert),(255,0,0),1)
        img2 = cv2.line(img2,(tbl_lower_horiz,tbl_lower_vert),(tbl_lower_horiz,tbl_upper_vert),(255,0,0),1)
        img2 = cv2.line(img2,(tbl_upper_horiz,tbl_lower_vert),(tbl_upper_horiz,tbl_upper_vert),(255,0,0),1)

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

    # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Frame", 960,540)

    scale_percent = 75 # percent of original size
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


# print (pts_global)
# print (dst_global)

# plt.xlim(0, 2)
# plt.ylim(1.5, 0)
# plt.show()



position = np.array(position)
position = position.reshape(-1,2)
heading = np.array(heading)
heading = heading.reshape(-1,1)
reading_time = np.array(reading_time)
reading_time = reading_time.reshape(-1,1)
fps_array = np.array(fps_array)
fps_array = fps_array.reshape(-1,1)
# print (position)

x, y = position.T

# fig, plt2 = plt.subplots()
# plt.figure()
# plt.plot(x, y)

# plt.xlim(0, 2)
# plt.ylim(1.5, 0)

# plt.set(xlabel='Width (m)', ylabel='Height (m)',
#        title='Position')
# plt.grid()

plt.plot(x, y)
plt.xlim(0, 2000)
plt.ylim(0, 1500)
# plt.set(xlabel='Width (m)', ylabel='Height (m)', title='Position')
plt.xlabel('Width (m)')
plt.ylabel('Height (m)')
plt.title('Position')
plt.grid()
# plt.add_patch(arrow)
# plt.arrow(cX,cY,AC_scaled[0],AC_scaled[1],head_width=(sf/2), head_length=(sf/2), fc='k', ec='k')
# plt.savefig("test.png")
plt.show()

print("Time taken:", datetime.now() - startTime)
print("Start position:", position[0])
print("Finish position:", position[(len(position) - 1)])


# cap.release()
# cv2.destroyAllWindows()
timestr = time.strftime("%Y%m%d-%H%M%S")
position_plus_heading = np.concatenate((reading_time, position, heading, fps_array), axis=1)

np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "pts_ORB_" + timestr + ".csv"), position_plus_heading, delimiter=",")

# np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "pts.csv"), pts_global, delimiter=",")
# np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "dst.csv"), dst_global, delimiter=",")