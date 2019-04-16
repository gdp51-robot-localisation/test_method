import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Define the codec and create VideoWriter object

# fourcc = cv.VideoWriter_fourcc(*'XVID')
# fourcc = cv.VideoWriter_fourcc(*'h264')
fourcc = cv.VideoWriter_fourcc(*'X264')
# fourcc = cv.FOURCC('8', 'B', 'P', 'S')
# out = cv.VideoWriter('output_h264_30.mov',fourcc, 20.0, (1920,1080))
# out = cv.VideoWriter('output_zeros.avi',-1, 30, (1920,1080))
out = cv.VideoWriter('output_mkv_20.mkv',fourcc, 20.0, (1920,1080))

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