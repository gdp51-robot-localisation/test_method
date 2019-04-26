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
hessian_input = 2700
surf = cv2.xfeatures2d.SURF_create(hessianThreshold = hessian_input, extended = 1)
# upright = false

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("output_QT_1080.mov")

# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(img1,None)

pts_global = []
dst_global = []

position = []
heading = []
reading_time = []
fps_array = []
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

# tbl_lower_horiz = 0
# tbl_upper_horiz = 1920
# tbl_lower_vert = 0
# tbl_upper_vert = 1080
# tbl_lower_horiz = 340
# tbl_upper_horiz = 1550
# tbl_lower_vert = 85
# tbl_upper_vert = 1000

# VALIDATION RUN
# tbl_lower_horiz = 346.1780369763
# tbl_upper_horiz = 1537.0186928970
# tbl_lower_vert = 92.8811144128
# tbl_upper_vert = 987.6189652272

# tl = (np.float32(tbl_lower_vert),np.float32(tbl_lower_horiz))
# tr = (np.float32(tbl_lower_vert),np.float32(tbl_upper_horiz))
# br = (np.float32(tbl_upper_vert),np.float32(tbl_upper_horiz))
# bl = (np.float32(tbl_upper_vert),np.float32(tbl_lower_horiz))

# tl = (np.float32(tbl_lower_horiz),np.float32(tbl_lower_vert))
# tr = (np.float32(tbl_upper_horiz),np.float32(tbl_lower_vert))
# br = (np.float32(tbl_upper_horiz),np.float32(tbl_upper_vert))
# bl = (np.float32(tbl_lower_horiz),np.float32(tbl_upper_vert))
# tl = np.float32([341.24725,87.78945])
# tr = np.float32([1541.55817,88.25892])
# br = np.float32([1543.52226,990.32708])
# bl = np.float32([339.34469,990.79555])

# Warp test
# bl = np.float32([335.05152,991.53404])
# br = np.float32([1539.39872,990.41745])
# tl = np.float32([338.22159,87.75624])
# tr = np.float32([1539.12241,87.43325])

# Warp test
# bl = np.float32([335.68484497,990.78814697])
# br = np.float32([1540.40332031,990.75927734])
# tl = np.float32([339.5017395,88.57030487])
# tr = np.float32([1539.82580566,89.25173187])

# # Warp and undistort test
# bl = np.float32([342.501,986.6])
# br = np.float32([1536.7289,987.73251])
# tl = np.float32([346.78525,93.52340])
# tr = np.float32([1534.49227,92.25574])
# bl = np.float32([341.90270996,986.7958374])
# # 
# bl = np.float32([341.70907593,987.79229736])
# br = np.float32([1539.68701172,990.0802002])
# tl = np.float32([346.43539429,92.95640564])
# tr = np.float32([1538.23413086,91.81403351])

# # Robot test 24/04/19
# bl = np.float32([283.0,1028.0])
# br = np.float32([1596.3,1030.2])
# tl = np.float32([286.5,46.7])
# tr = np.float32([1595.3,44.3])

# # Robot test 2YG7 25/04/19
bl = np.float32([177,1118])
br = np.float32([1693,1124.12045])
tl = np.float32([182.8,-48])
tr = np.float32([1687,-65])

rect = (tl, tr, bl, br)
# print(rect)
rect = np.array(rect)
# print(rect)

# # compute the width of the new image, which will be the
# # maximum distance between bottom-right and bottom-left
# # x-coordiates or the top-right and top-left x-coordinates
# widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
# widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
# maxWidth = max(int(widthA), int(widthB))
# print(maxWidth)

maxWidth=1200
maxHeight= round(0.75 * maxWidth)

# # compute the height of the new image, which will be the
# # maximum distance between the top-right and bottom-right
# # y-coordinates or the top-left and bottom-left y-coordinates
# heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
# heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
# maxHeight = max(int(heightA), int(heightB))
# print(maxHeight)

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


# cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Frame", 600,350)

while True:
    _, img2 = cap.read()
    img2 = cv2.undistort(img2, mtx_1, dist_1)
    img2 = cv2.warpPerspective(img2, M_persp, (maxWidth, maxHeight))
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Start timer
    timer = cv2.getTickCount()

    # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1,None)
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
        # cX = float(((dst[2,0,0] - dst[0,0,0])/2.0) + dst[0,0,0])
        # cY = float(((dst[2,0,1] - dst[0,0,1])/2.0) + dst[0,0,1])
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
        # print (cX)
        # print (cY)

        # cX = [a - tbl_lower_horiz for a in x]
        # cX = [(2 * (a / (tbl_upper_horiz - tbl_lower_horiz))) for a in x]
        # cY = [a - tbl_lower_vert for a in y]
        # cY = [(1.5 * (a / (tbl_upper_vert - tbl_lower_vert))) for a in y]

        # cX = 2000 * ((cX) / (tbl_upper_horiz - tbl_lower_horiz))
        # cY = 1500 - (1500 * ((cY) / (tbl_upper_vert - tbl_lower_vert)))

        cX = 2000 * (cX / maxWidth)
        cY = 1500 - (1500 * (cY / maxHeight))

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
        # img2 = cv2.line(img2,(round(tbl_lower_horiz),round(tbl_lower_vert)),(round(tbl_upper_horiz),round(tbl_lower_vert)),(255,0,0),1)
        # img2 = cv2.line(img2,(round(tbl_lower_horiz),round(tbl_upper_vert)),(round(tbl_upper_horiz),round(tbl_upper_vert)),(255,0,0),1)
        # img2 = cv2.line(img2,(round(tbl_lower_horiz),round(tbl_lower_vert)),(round(tbl_lower_horiz),round(tbl_upper_vert)),(255,0,0),1)
        # img2 = cv2.line(img2,(round(tbl_upper_horiz),round(tbl_lower_vert)),(round(tbl_upper_horiz),round(tbl_upper_vert)),(255,0,0),1)

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
plt.xlabel('Width (mm)')
plt.ylabel('Height (mm)')
plt.title('Position')
plt.grid()
# plt.add_patch(arrow)
# plt.arrow(cX,cY,AC_scaled[0],AC_scaled[1],head_width=(sf/2), head_length=(sf/2), fc='k', ec='k')
# plt.savefig("test.png")
plt.show()

print("Time taken:", datetime.now() - startTime)
print("Start position:", position[0])
print("Finish position:", position[(len(position) - 1)])
print("Mean position:", np.mean(position,0))


# cap.release()
# cv2.destroyAllWindows()
timestr = time.strftime("%Y%m%d-%H%M%S")
position_plus_heading = np.concatenate((reading_time, position, heading, fps_array), axis=1)

np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "pts_SURF_" + timestr + ".csv"), position_plus_heading, delimiter=",")

# np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "pts.csv"), pts_global, delimiter=",")
# np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "dst.csv"), dst_global, delimiter=",")