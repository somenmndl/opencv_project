# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:13:56 2023

@author: 91736
"""

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path= "C:\Face Recognition + Attendance\Images"
images=[]
classNames=[]
myList= os.listdir(path)
#print(myList)
for cls in myList:
    curImg= cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown= findEncodings(images)
#print(len(encodeListKnown))

def markAttendance(name):
    #r+ to read and write at the same time
    with open("C:\Face Recognition + Attendance\Attendance.csv",'r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry= line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now= datetime.now()
            dtString= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


cap=cv2.VideoCapture(0)
while True:
    success,img= cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    facesCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame= face_recognition.face_encodings(imgS,facesCurFrame)
    
    #We are using zip bcoz we want both to work in the same loop 
    for encodeFace, faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches= face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis= face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)
        
        if matches[matchIndex]:
            name= classNames[matchIndex].upper()
            print(name)
            
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4 
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-30),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+5,y2-5),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
            
    cv2.imshow("WebCam", img)
    key=cv2.waitKey(1)
    if key==27:
        break
cv2.release()
cv2.destroyAllWindows()
    
            
