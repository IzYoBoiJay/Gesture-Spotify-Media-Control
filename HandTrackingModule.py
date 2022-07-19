# -------------------- Packages needed for Computer Vision ------------------- #
import cv2 #OpenCV
import numpy


import mediapipe #Mediapipe Framework for Hand-Tracking

import time
previousTime = 0

# ------------------------- Camera Parameters & Setup ------------------------ #
cameraID = 0
cameraWidthPixels = 1280
cameraHeightPixels = 720

#Setup camera capture to the video capture of specified camera ID
cameraCapture = cv2.VideoCapture(cameraID)

#Set width, or 3, to the width pixels
cameraCapture.set(3, cameraWidthPixels)

#Set height, or 4, to the height pixels
cameraCapture.set(4, cameraHeightPixels)

# ---------------------------- FPS Text Parameters --------------------------- #

#Bottom-Left Coordinates of the Text String onto the Image
org = (40, 50)

#Font
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontScale = 2

lineThickness = 3
lineType = cv2.LINE_AA

#Color in BGR format
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# ---------------------------------------------------------------------------- #
#                                   Mediapipe                                  #
# ---------------------------------------------------------------------------- #

class handDetector():

    def __init__(self, staticImageMode = False, maxNumHands = 2, modelComplexity = 1, minDetectConfidence = 0.5, minTrackingConfidence = 0.5):

        self.staticImageMode = staticImageMode
        self.maxNumHands = maxNumHands
        self.modelComplexity = modelComplexity
        self.minDetectConfidence = minDetectConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mediapipeHands = mediapipe.solutions.hands
        self.hands = self.mediapipeHands.Hands(self.staticImageMode, self.maxNumHands, self.modelComplexity, self.minDetectConfidence, self.minTrackingConfidence)
        self.mediapipeDraw = mediapipe.solutions.drawing_utils  

    def detectHands(self, image, drawHands = True):

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:

            for handLandmarks in self.results.multi_hand_landmarks:

                if drawHands:

                    self.mediapipeDraw.draw_landmarks(image, handLandmarks, self.mediapipeHands.HAND_CONNECTIONS)

        return image

    def handPointPositions(self, image, handNum = 0, drawPoints = True):

        landmarkList = []

        if self.results.multi_hand_landmarks:

            hand = self.results.multi_hand_landmarks[handNum]

            for id, landmark in enumerate(hand.landmark):

                height, width, channels = image.shape

                xCoord, yCoord = int(landmark.x * width), int(landmark.y * height)

                landmarkList.append([id, xCoord, yCoord])

            if drawPoints:
                cv2.circle(image, (xCoord, yCoord), 15, GREEN, cv2.FILLED)

        return landmarkList


# ---------------------------------------------------------------------------- #
#                               Utility Functions                              #
# ---------------------------------------------------------------------------- #
def calculateFPS():

    currentTime = time.time()
    global previousTime

    fps = 1 / (currentTime - previousTime)

    previousTime = currentTime

    return fps

# ---------------------------------------------------------------------------- #
#                                 Main Function                                #
# ---------------------------------------------------------------------------- #
def main():

    handTracker = handDetector()

    while True:
        
        #Let the image be what the camera captures
        success, image = cameraCapture.read()

        image = handTracker.detectHands(image)
        landmarkList = handTracker.handPointPositions(image)

        if len(landmarkList) != 0:

            print(landmarkList[4])

        cv2.putText(image, 'FPS: {}'.format(int(calculateFPS())), org, font, fontScale, BLACK, lineThickness, lineType)

        #imshow takes a window name and the image to display. Automatically fits to image size.
        cv2.imshow("Spotify Controls", image)

        #Waits for user to press any key with a 1ms Delay
        cv2.waitKey(1)

if __name__ == "__main__":
    main()