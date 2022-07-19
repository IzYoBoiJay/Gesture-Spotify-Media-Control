# -------------------- Packages needed for Computer Vision ------------------- #
import cv2
import numpy

import time
previousTime = 0

# ----------------------------- Camera Parameters ---------------------------- #
cameraWidthPixels = 1280
cameraHeightPixels = 720
cameraID = 0

# ------------------------------- Camera Setup ------------------------------- #
cameraCapture = cv2.VideoCapture(cameraID)

#Set width, or 3, to the width pixels
cameraCapture.set(3, cameraWidthPixels)

#Set height, or 4, to the height pixels
cameraCapture.set(4, cameraHeightPixels)

def calculateFPS():

    currentTime = time.time()
    global previousTime

    fps = 1 / (currentTime - previousTime)

    previousTime = currentTime

    return fps

while True:
    
    #Let the image be what the camera captures
    success, image = cameraCapture.read()

    cv2.putText(image, 'FPS: {}'.format(int(calculateFPS())), (40, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 3)

    #imshow takes a window name and the image to display. Automatically fits to image size.
    cv2.imshow("Spotify Controls", image)

    #Waits for user to press any key with a 1ms Delay
    cv2.waitKey(1)