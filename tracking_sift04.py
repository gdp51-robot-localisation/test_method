import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os, os.path
import math

MIN_MATCH_COUNT = 10

img1 = cv2.imread('Pattern3_small.jpg',0)          # queryImage
# img1 = cv2.imread('Pattern_robot.png',0)          # queryImage
# img2 = cv2.imread('Pattern4.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("output_x264.mp4")

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)

pts_global = []
dst_global = []

position = []
heading = []
# plt.axis([0, 1280, 0, 720])

tbl_upper_horiz = 1539
tbl_lower_horiz = 343
tbl_upper_vert = 1008
tbl_lower_vert = 110

# cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Frame", 600,350)

while True:
    _, img2 = cap.read()

    # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

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

        cX = 2 * ((cX - tbl_lower_horiz) / (tbl_upper_horiz - tbl_lower_horiz))
        cY = 1.5 * ((cY - tbl_lower_vert) / (tbl_upper_vert - tbl_lower_vert))

        position.extend([cX,cY])
        heading.append(robot_angle)

        sf = 0.04
        scale_factor = sf/AC_length
        
        AC_scaled = np.multiply(AC, scale_factor)
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
        img2 = cv2.putText(img2, "Heading: " + str(int(robot_angle)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        img2 = cv2.line(img2,(tbl_lower_horiz,tbl_lower_vert),(tbl_upper_horiz,tbl_lower_vert),(255,0,0),1)
        img2 = cv2.line(img2,(tbl_lower_horiz,tbl_upper_vert),(tbl_upper_horiz,tbl_upper_vert),(255,0,0),1)
        img2 = cv2.line(img2,(tbl_lower_horiz,tbl_lower_vert),(tbl_lower_horiz,tbl_upper_vert),(255,0,0),1)
        img2 = cv2.line(img2,(tbl_upper_horiz,tbl_lower_vert),(tbl_upper_horiz,tbl_upper_vert),(255,0,0),1)

    else:
        # print ("Not enough matches are found - %d/%d") % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

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
        
    # cap.release()
    # cv2.destroyAllWindows()



# print (pts_global)
# print (dst_global)

# plt.xlim(0, 2)
# plt.ylim(1.5, 0)
# plt.show()



position = np.array(position)
position = position.reshape(-1,2)
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
plt.xlim(0, 2)
plt.ylim(1.5, 0)
# plt.set(xlabel='Width (m)', ylabel='Height (m)', title='Position')
plt.xlabel('Width (m)')
plt.ylabel('Height (m)')
plt.title('Position')
plt.grid()
# plt.add_patch(arrow)
# plt.arrow(cX,cY,AC_scaled[0],AC_scaled[1],head_width=(sf/2), head_length=(sf/2), fc='k', ec='k')
# plt.savefig("test.png")
plt.show()




cap.release()
cv2.destroyAllWindows()

np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "pts.csv"), position, delimiter=",")

# np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "pts.csv"), pts_global, delimiter=",")
# np.savetxt(os.path.join("/Users/michaelleat/Documents/Education/MechEng/Year4/GDP/TestMethod/Code/01", "dst.csv"), dst_global, delimiter=",")