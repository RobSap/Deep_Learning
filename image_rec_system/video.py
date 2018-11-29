import numpy as np
import cv2
import sys

#Video tutorial
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

#To download the xml files
# https://github.com/opencv/opencv/tree/master/data/haarcascades



cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
full_body = cv2.CascadeClassifier('haarcascade_fullbody.xml')
upper_body = cv2.CascadeClassifier('haarcascade_upperbody.xml')

if (full_body.empty()):
    print("Error finding full boyd cascade file")

elif eye_cascade.empty():
    print("Error finding eye cascade file")

elif face_cascade.empty():
    print("Error finding face cascade file")

elif upper_body.empty():
    print("Error finding face cascade file")

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    #Make screen smaller for better frame rate
    frame = cv2.resize(frame,(640,360))
    frame= cv2.flip( frame, 1 )
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #print(str(faces) +" " + str(type(faces)))
    #if isinstance(faces, tuple):
    #    print(1)
    #if isinstance(faces, np.ndarray):
    #   print(2)
    if not isinstance(faces, tuple):
        print("Send an alert, person detected")
 
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    
    body = full_body.detectMultiScale(gray,1.1,1 ) 
    
    for (x,y,w,h) in body:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    upper = upper_body.detectMultiScale(gray,1.1,1 )
    
    for (x,y,w,h) in upper:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    #Display the face and eyes
    cv2.imshow('img',frame)

    # Display the resulting frame from video (gray)
    #cv2.imshow('frame',gray)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
