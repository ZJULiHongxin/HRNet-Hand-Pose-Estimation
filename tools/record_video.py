# coding:utf-8
import sys
import cv2
import numpy as np
# 选取摄像头，0为笔记本内置的摄像头，1,2···为外接的摄像头
cap = cv2.VideoCapture(1)
# cap.set(3,1080)
# cap.set(4,720)

# 为保存视频做准备
fourcc = cv2.VideoWriter_fourcc(*"DIVX")
# 第三个参数则是镜头快慢的，20为正常，小于二十为慢镜头
out = cv2.VideoWriter('./output.avi', fourcc, 30, (640,480))
cv2.waitKey(2000)
count = 0
while count<500:
    # 一帧一帧的获取图像
    ret,frame = cap.read()
    count+=1
    if ret == True:
        #frame = cv2.flip(frame, 1)
        # 在帧上进行操作
        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 开始保存视频
        out.write(frame)
        # 显示结果帧
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

# 释放摄像头资源
cap.release()
out.release()
cv2.destroyAllWindows()