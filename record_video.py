import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Define the codec and create VideoWriter object

fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('output.mp4',fourcc, 25.0, (1920,1080))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv.flip(frame,0)
        # write the flipped frame
        out.write(frame)
        cv.imshow('frame',frame)

        key = cv.waitKey(1)
        if key == 27:   #ESC
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()