import cv2 as cv

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
sys.path.append(ROOT_DIR)

capture = cv.VideoCapture(0)  # to open Camera

xml_file_path = os.path.join(ROOT_DIR, 'haarcascade_frontalface_default.xml')

# accessing pretrained model
pretrained_model = cv.CascadeClassifier(xml_file_path)

while True:
    boolean, frame = capture.read()
    if boolean:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        coordinate_list = pretrained_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        # drawing rectangle in frame
        for (x, y, w, h) in coordinate_list:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Display detected face
        cv.imshow("Real time Face Detection", frame)

        # condition to break out of while loop
        if cv.waitKey(20) == ord('x'):
            break

capture.release()
cv.destroyAllWindows()