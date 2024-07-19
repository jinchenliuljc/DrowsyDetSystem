# %%writefile sttest.py

import cv2
import numpy as np
import dlib
import streamlit as st
import time


def calc_dist(dot1,dot2):
    dist = abs(dot1.x-dot2.x)+abs(dot1.y-dot2.y)
    return dist

def mouth_open(dots,thresh):
    dist1 = calc_dist(dots.part(61),dots.part(67))
    dist2 = calc_dist(dots.part(63),dots.part(65))
    dist3 = calc_dist(dots.part(48),dots.part(54))
    EAR = (dist1+dist2)/(2*dist3)
    if EAR>thresh:
        return True
    else:    
        return False

def blink(dots,thresh):
    dist1 = calc_dist(dots.part(37),dots.part(41))
    dist2 = calc_dist(dots.part(38),dots.part(40))
    dist3 = calc_dist(dots.part(36),dots.part(39))
    dist4 = calc_dist(dots.part(43),dots.part(47))
    dist5 = calc_dist(dots.part(44),dots.part(46))
    dist6 = calc_dist(dots.part(42),dots.part(45))
    EAR1 = (dist1+dist2)/(2*dist3)
    EAR2 = (dist4+dist5)/(2*dist6)
    if EAR1<thresh and EAR2<thresh:
        return True
    else:    
        return False
    
def adjust_yawn_thresh(x):
    global num_yawns_thresh
    num_yawns_thresh = x
    
def adjust_blink_thresh(x):
    global num_blinks_thresh
    num_blinks_thresh = x

    
def reset(event,x,y,flags,param):
    global num_blink, num_yawn, yawn, time1, warning
    if event==cv2.EVENT_LBUTTONDOWN:
        num_blink = 0
        num_yawn = 0
        yawn = 0
        time1 = 0
        warning = False
    
    
    
# 加载dlib 人脸检测器
detector = dlib.get_frontal_face_detector()
# 加载dlib 人脸关键点
predictor = dlib.shape_predictor('weights/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat')

thresh_mouth = 0.3
thresh_eyes = 0.2
num_yawns_thresh = 5
num_blinks_thresh = 20
max_times = 5

# 打开摄像头
cap = cv2.VideoCapture(0)
num_blink = 0
num_yawn = 0
yawn = 0
time1 = time.time()
warning = False

flag, frame = cap.read()
cv2.imshow('face',frame)
cv2.createTrackbar('thresh_yawns','face',5,10,adjust_yawn_thresh)
cv2.createTrackbar('thresh_blinks','face',20,50,adjust_blink_thresh)

while(1):
    flag, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #b, g, r = cv2.split(frame)
    #frame_RGB = cv2.merge((r, g ,b))
    rets = detector(frame_gray, 0)
    for face in rets:
        dots = predictor(frame_gray, face)
        
        if not warning==True:
            criteria_mouth = mouth_open(dots,thresh_mouth)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if criteria_mouth:
                frame = cv2.putText(frame, 'Yawn', (50, 300), font, 1.2, (255, 255, 255), 2)
                num_yawn += 1
            else:
                frame = cv2.putText(frame, "Normal", (50, 300), font,1.2, (255, 255, 255), 2)
                num_yawn = 0
            
#             frame = cv2.putText(frame, f"{num_yawn}", (50, 400), font,1.2, (255, 255, 255), 2)
            if num_yawn > 20:
                yawn += 1
                num_yawn = 0
                
            frame = cv2.putText(frame, f"Num Yawn:{yawn}", (50, 400), font,1.2, (255, 255, 255), 2)
            criteria_eyes = blink(dots,thresh_eyes)
            if criteria_eyes:
    #             frame = cv2.putText(frame, 'Yaw', (50, 300), font, 1.2, (255, 255, 255), 2)
                num_blink+=1
            else:
                num_blink=0
            frame = cv2.putText(frame, f"Num Frame Eye-closed:{num_blink}", (50, 100), font,1.2, (255, 255, 255), 2)
            time2 = time.time()
            interval = time2-time1
            if interval > 60:
                time1 = time.time()
                num_blink = 0
                yawn = 0

            if yawn > num_yawns_thresh or num_blink>num_blinks_thresh:
                warning=True
            frame = cv2.putText(frame, f"{int(time2-time1)}", (50, 200), font,1.2, (255, 255, 255), 2)
        else:
            frame = cv2.putText(frame, "Please take a rest!!!", (0, 200), font,2, (255, 255, 255), 2)
            cv2.setMouseCallback("face",reset)
        for i in dots.parts():
            pos_dot = (i.x, i.y)
            frame_face = cv2.circle(frame, pos_dot, 1, (0,255,0), 2)

        cv2.imshow('face', frame_face)
#         print(thresh_mouth)
#         cv2.createTrackbar('g','show',0,255,nothing)
#         cv2.createTrackbar('b','show',0,255,nothing)
#     with st.empty():
#         st.image(frame_face)
    k = cv2.waitKey(1)
    if k & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
