  import cv2
import sys
from datetime import datetime
import pandas
import csv
import time
import json
#import urllib
import urllib.request
import base64
from flask import request

bodyCascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
video_capture=cv2.VideoCapture(0)
count=0
time = []

static_back = None

motion_list = [ None, None ]
df = pandas.DataFrame(columns = ["Start", "End","Status"])

while True:
   
    ret, frame = video_capture.read()
    motion = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    body = bodyCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100,200)
    )
    if static_back is None:
        static_back = gray
        continue
 
    # Difference between static background 
    # and current frame(which is GaussianBlur)
    diff_frame = cv2.absdiff(static_back, gray)
 
    # If change in between static background and
    # current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    
    (_, cnts, _) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1

        
    motion_list.append(motion)
 
    motion_list = motion_list[-2:]
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.now())
 
    # Appending End time of motion
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.now())
   
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if count<5:
            cv2.imwrite("C:\\Users\\Admin\\Desktop\\hackathon\\detect\\image\\img"+str(count)+".jpg",frame)
        count=count+1
    cv2.imshow('Video', frame)

    
    key = cv2.waitKey(1)
    if key == ord('q'):
        if motion == 1:
            time.append(datetime.now())
        break
        
for i in range(0, len(time), 2):
    df = df.append({"Start":time[i], "End":time[i + 1], "Status":motion}, ignore_index = True)
        
df.to_csv("C:\\Users\\Admin\\Desktop\\hackathon\\detect\\Status.csv")
csvfile = open('Status.csv', 'r')
jsonfile = open('Body.json', 'w')

fieldnames = ("Start","End","Status")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')

jsonfile = open('C:\\Users\\Admin\\Desktop\\hackathon\\detect\\image.json', 'w')    
image=cv2.imread("C:\\Users\\Admin\\Desktop\\hackathon\\detect\\image\\img0.jpg")
retval, buffer = cv2.imencode('.jpg', image)
fieldnames = ("image")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    jpg_as_text = base64.b64encode(buffer)
    json.dump(row, jpg_as_text)
    

    
video_capture.release()
cv2.destroyAllWindows()
