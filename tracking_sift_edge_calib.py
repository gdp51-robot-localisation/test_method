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

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# upright = false

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("output_QT_1080.mov")

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)

pts_global = []
dst_global = []

position = []
heading = []
reading_time = []
fps_array = []
pattern_width = []
BL = []
BR = []
TL = []
TR = []
# plt.axis([0, 1280, 0, 720])

mtx_1 = np.array([[7615.086684842451, 0.0, 934.6619126632753],[0.0, 7589.141593519471, 560.8448319169089],[0.0, 0.0, 1.0]])
dist_1 = np.array([3.3036628821574143, -284.6056262111876, -0.00990095676339995, -0.01422899829406913, -3.8589533787510892])

# tbl_lower_horiz = 288.9509016906 #344.3218434916 #343
# tbl_upper_horiz = 1597.5829711800 #1533.3151804436 #1539
# tbl_lower_vert = 49.2866110651 #95.2119867521 #110
# tbl_upper_vert = 1030.9299197763 #986.8746805046 #1008

# tbl_lower_horiz = 344.3218434916 #343
# tbl_upper_horiz = 1533.3151804436 #1539
# tbl_lower_vert = 95.2119867521 #110
# tbl_upper_vert = 986.8746805046 #1008

tbl_lower_horiz = 0
tbl_upper_horiz = 1920
tbl_lower_vert = 0
tbl_upper_vert = 1080



# tbl_lower_horiz = 346.1780369763
# tbl_upper_horiz = 1537.0186928970
# tbl_lower_vert = 92.8811144128
# tbl_upper_vert = 987.6189652272


# cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Frame", 600,350)

while True:
    _, img2 = cap.read()
    img2 = cv2.undistort(img2, mtx_1, dist_1)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Start timer
    timer = cv2.getTickCount()

    # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # store all the good matches as per Lowe's ratio test.
    good = []
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)

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

        dst_bl = dst[1]
        dst_tl = dst[0]
        dst_br = dst[2]
        dst_tr = dst[3]

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
        cX1 = float(((dst[2,0,0] - dst[0,0,0])/2.0) + dst[0,0,0])
        # print(cX1)
        cY1 = float(((dst[2,0,1] - dst[0,0,1])/2.0) + dst[0,0,1])
        # print(cY1)
        cX2 = float(((dst[3,0,0] - dst[1,0,0])/2.0) + dst[1,0,0])
        # print(cX2)
        cY2 = float(((dst[3,0,1] - dst[1,0,1])/2.0) + dst[1,0,1])
        # print(cY2)
        cX = (cX1 + cX2)/2.0
        cY = (cY1 + cY2)/2.0

        # cX = float(((dst[2,0,0] - dst[0,0,0])/2.0) + dst[0,0,0])
        # cY = float(((dst[3,0,1] - dst[1,0,1])/2.0) + dst[0,0,1])
        
        cX_delta = float((dst[2,0,0] - dst[0,0,0]))
        cY_delta = float((dst[2,0,1] - dst[0,0,1]))
        # print (cX)
        # print (cY)
        # BL = 

        # cX = [a - tbl_lower_horiz for a in x]
        # cX = [(2 * (a / (tbl_upper_horiz - tbl_lower_horiz))) for a in x]
        # cY = [a - tbl_lower_vert for a in y]
        # cY = [(1.5 * (a / (tbl_upper_vert - tbl_lower_vert))) for a in y]

        # cX = 2000 * ((cX - tbl_lower_horiz) / (tbl_upper_horiz - tbl_lower_horiz))
        # cY = 1500 - (1500 * ((cY - tbl_lower_vert) / (tbl_upper_vert - tbl_lower_vert)))

        position.extend([cX,cY])
        pattern_width.extend([cX_delta,cY_delta])
        heading.append(robot_angle)
        time_delta = datetime.now() - startTime
        seconds_reading = time_delta.total_seconds()
        reading_time.append(seconds_reading)
        fps_array.append(fps)
        BL.extend(dst_bl)
        BR.extend(dst_br)
        TL.extend(dst_tl)
        TR.extend(dst_tr)


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
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,1, cv2.LINE_AA)
        img2 = cv2.line(img2,(round(tbl_lower_horiz),round(tbl_lower_vert)),(round(tbl_upper_horiz),round(tbl_lower_vert)),(255,0,0),1)
        img2 = cv2.line(img2,(round(tbl_lower_horiz),round(tbl_upper_vert)),(round(tbl_upper_horiz),round(tbl_upper_vert)),(255,0,0),1)
        img2 = cv2.line(img2,(round(tbl_lower_horiz),round(tbl_lower_vert)),(round(tbl_lower_horiz),round(tbl_upper_vert)),(255,0,0),1)
        img2 = cv2.line(img2,(round(tbl_upper_horiz),round(tbl_lower_vert)),(round(tbl_upper_horiz),round(tbl_upper_vert)),(255,0,0),1)

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
pattern_width = np.array(pattern_width)
pattern_width = pattern_width.reshape(-1,2)
heading = np.array(heading)
heading = heading.reshape(-1,1)
reading_time = np.array(reading_time)
reading_time = reading_time.reshape(-1,1)
fps_array = np.array(fps_array)
fps_array = fps_array.reshape(-1,1)
BL = np.array(BL)
BL = BL.reshape(-1,2)
BR = np.array(BR)
BR = BR.reshape(-1,2)
TL = np.array(TL)
TL = TL.reshape(-1,2)
TR = np.array(TR)
TR = TR.reshape(-1,2)
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

# plt.plot(x, y)
# plt.xlim(0, 2000)
# plt.ylim(0, 1500)
# # plt.set(xlabel='Width (m)', ylabel='Height (m)', title='Position')
# plt.xlabel('Width (m)')
# plt.ylabel('Height (m)')
# plt.title('Position')
# plt.grid()
# # plt.add_patch(arrow)
# # plt.arrow(cX,cY,AC_scaled[0],AC_scaled[1],head_width=(sf/2), head_length=(sf/2), fc='k', ec='k')
# # plt.savefig("test.png")
# plt.show()

print("Time taken:", datetime.now() - startTime)
print("Start position:", position[0])
print("Finish position:", position[(len(position) - 1)])
print("Mean position:", np.mean(position,0))
print("Mean pattern width:", np.mean(pattern_width,0))
print("Mean BL:", np.mean(BL,0))
print("Mean BR:", np.mean(BR,0))
print("Mean TL:", np.mean(TL,0))
print("Mean TR:", np.mean(TR,0))


# cap.release()
# cv2.destroyAllWindows()
# timestr = time.strftime("%Y%m%d-%H%M%S")
# position_plus_heading = np.concatenate((reading_time, position, heading, fps_array), axis=1)

# np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "pts_SURF_H" + str(hessian_input) + "_" + timestr + ".csv"), position_plus_heading, delimiter=",")

# np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "pts.csv"), pts_global, delimiter=",")
# np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "dst.csv"), dst_global, delimiter=",")