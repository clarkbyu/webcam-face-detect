# webcam_face_detection.py
"""
Webcam Face Detection

@author: Clark Brown
@date: 21 March 2020

Based on the tutorial code at https://github.com/DilanKalpa/FaceDetection
"""

import cv2
import os
from datetime import datetime

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

if __name__ == "__main__":
    print("\nWebcam Face Detection:")
    print("\tPress SPACE to capture image. Press ESC to exit.")

    # Set video source as default webcam
    video_capture = cv2.VideoCapture(0)

    img_counter = 0
    capture_dir = 'captures/' + datetime.now().strftime('%Y%m%d%H%M%S')
    os.mkdir(capture_dir)

    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        k = cv2.waitKey(1)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Add rectangle around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the new frame
        cv2.imshow('Face Detection', frame)

        # Key listeners
        if k%256 == 27:
            # ESC Pressed
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = capture_dir + "/facedetect_webcam_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    # Release capture, clean up
    if img_counter == 0:
        os.rmdir(capture_dir)
    video_capture.release()
    cv2.destroyAllWindows()