import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = "images"
images = [] # we will get the images in this list
classNames = [] # we will get the names of images in this list

myList = os.listdir(path)
print(myList)

for cls in myList:
    curImg = cv2.imread(path + "/" + cls)   
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

"""
next we will make a simple function which will find the encodings for us for every image
in the "images" list.
"""

def findEncodings(images):
    """
    this function will require a list of images
    """
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    """
    if the person is already there in attendance list, she/he should not be included 
    again in the attendance list. if the person is not already present, this function 
    will append the name and timing of the person entry into the video frame to the 
    Attendance.csv file.
    """
    with open("Attendance.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtstring}")


    


encodeListKnown = findEncodings(images)
# print(encodeListKnown)
# print(len(encodeListKnown))
print("encoding complete")

"""
now we will match the encodings. but we don't have the image to match with, so let's
initialise our webcam.
"""

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) # one-fourth of the original size
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    """
    we might find multiple faces in the video, so we will send the location of faces
    for the encoding process
    """
    faces_in_current_frame = face_recognition.face_locations(imgS)
    encodesCurrentFrame = face_recognition.face_encodings(imgS, faces_in_current_frame)
    """
    as in a video captured frame, we can have multiple pictures, so at the time of encoding
    we will pass the location of these faces too. previously when we used encoding function
    we did not pass the face location as we had only single face in the picture, but here
    there can be many so we did pass the face location.
    """
    for encodeFace, faceLoc in zip(encodesCurrentFrame, faces_in_current_frame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        match_index = np.argmin(faceDis)
        # print(classNames[match_index])
        if matches[match_index]:
            name = classNames[match_index].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y1-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            markAttendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)




