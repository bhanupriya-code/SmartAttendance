import cv2
import numpy as npy
import face_recognition as face_rec
import os
from datetime import datetime

def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

path = 'employee_images'
employeeimg = []
employeename = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}\{cl}')
    employeeimg.append(curImg)
    employeename.append(os.path.splitext(cl)[0])

#print(employeename)

def findEncoding(Images):
    encoding_list = []
    for img in Images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        encoding_list.append(encodeimg)
    return encoding_list

def MarkAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList :
            now = datetime.now()
            timestr = now.strftime('%H: %M')
            f.writelines(f'\n{name}, {timestr}')




encode_list = findEncoding(employeeimg)

vid = cv2.VideoCapture(0)

while True :
    success, frame = vid.read()
    Smaller_frame = cv2.resize(frame, (0,0), None, 0.25, 0.25)

    facesInframe = face_rec.face_locations(Smaller_frame)
    encodeFacesInFrame = face_rec.face_encodings(Smaller_frame, facesInframe)

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInframe) :
        matches = face_rec.compare_faces(encode_list, encodeFace)
        facedis = face_rec.face_distance(encode_list, encodeFace)
        print(facedis)
        matchIndex = npy.argmin(facedis)

        if matches[matchIndex]:
            name = employeename[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendance(name)

    cv2.imshow('video', frame)
    cv2.waitKey(1)
