import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import math
import matplotlib.pyplot as plt
import os
import time
import shutil
import base64
import pandas as pd
import xlwt

#先进行二分类，face_type:1是方脸，2是圆脸，3是鹅蛋脸，4是瓜子脸
face_type = 0
fang_probability = 0
yuan_probability = 0
edan_probability = 0
guazi_probability = 0



# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img_rd = cv2.imread("face/face picture/face_edan/edan84.jpg")
img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

rgb_img = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)   # opencv的颜色空间是BGR，需要转为RGB才能用在dlib中
dets = detector(rgb_img, 1)

# 检测到的人脸数量
num_faces = len(dets)
if num_faces == 0:
    print("Sorry, there were no faces found in '{}'".format('test/yang.jpg'))
    exit()

# 识别人脸特征点，并保存下来
faces = dlib.full_object_detections()
for det in dets:
    faces.append(predictor(rgb_img, det))

# 待会要写的字体
font = cv2.FONT_HERSHEY_SIMPLEX

################################################################人脸对齐###################################################################
images = dlib.get_face_chips(rgb_img, faces, size=500)
# 显示计数，按照这个计数创建窗口
image_cnt = 0
# 显示对齐结果
for image in images:
    image_cnt += 1
    cv_rgb_image = np.array(image).astype(np.uint8)  # 先转换为numpy数组
    cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)  # opencv下颜色空间为bgr，所以从rgb转换为bgr
    #cv2.imshow('%s' % (image_cnt), cv_bgr_image)

################################################################关键点标定###################################################################


cv_bgr_image_grey = cv2.cvtColor(cv_bgr_image, cv2.COLOR_RGB2GRAY)
dets_adjust = detector(cv_bgr_image_grey, 0)

if num_faces == 1:
    # 检测到一张人脸

    for i in range(num_faces):
        # 取特征点坐标
        landmarks = np.matrix([[p.x, p.y] for p in predictor(cv_bgr_image, dets_adjust[i]).parts()])
        cv2.rectangle(cv_bgr_image, (dlib.rectangle().left(), dlib.rectangle().top()), (dlib.rectangle().right(), dlib.rectangle().bottom()), (255, 255, 255))
        shape=predictor(cv_bgr_image,dlib.rectangle())
        for idx, point in enumerate(landmarks):
            # 68 点的坐标
            pos = (point[0, 0], point[0, 1])


            #利用 cv2.circle 给每个特征点画一个圈，共 68 个
            cv2.circle(cv_bgr_image, pos, 2, color=(139, 0, 0))
            #利用 cv2.putText 写数字 1-68
            cv2.putText(cv_bgr_image, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(cv_bgr_image, "faces: " + str(num_faces), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
else:
    # 没有检测到人脸
    cv2.putText(cv_bgr_image, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

################################################################脸型参数###################################################################
# # 绘出轮廓
# landmarks_face = landmarks[4:13, :]
# landmarks_face_x = np.array(landmarks_face)[:, 0]
# landmarks_face_y = np.array(landmarks_face)[:, 1]
# face_line = np.poly1d(np.polyfit(landmarks_face_x, landmarks_face_y, 4))  # 4存疑
#
# for t in range(landmarks_face[0, 0], landmarks_face[8, 0], 1):
#     y_face = np.int(face_line(t))
#     cv2.circle(cv_bgr_image, (t, y_face), 2, color=(139, 0, 0))

face_point1 = np.array(landmarks[1])
face_point2 = np.array(landmarks[15])
face_point3 = np.array(landmarks[8])
face_point4 = np.array(landmarks[27])
face_point5 = np.array(landmarks[4])
face_point6 = np.array(landmarks[12])
face_point7 = np.array(landmarks[6])
face_point8 = np.array(landmarks[10])
# faceup_x = (int(face_point1[:, 0]) + int(face_point2[:, 0])) / 2
# faceup_y = (int(face_point1[:, 1]) + int(face_point2[:, 1])) / 2
# landmarks = np.vstack((landmarks, np.matrix([[faceup_x, faceup_y]])))
# face_point4 = np.array(landmarks[68])

face_width = np.linalg.norm(face_point1 - face_point2)
face_height = np.linalg.norm(face_point3 - face_point4)
face_jaw_width = np.linalg.norm(face_point5 - face_point6)
face_shape_index = face_height/face_width
face_jaw_index = face_jaw_width/face_width

face_jawdown_width = np.linalg.norm(face_point7 - face_point8)
face_jaw_regression_index = face_jawdown_width / face_jaw_width


#以下数据来自matlab数据分析
if 101.2355 + 16.7486*face_shape_index - 142.0216*face_jaw_index > 0:
    edan_probability = 1
    guazi_probability = 1
    eg_probability = 1 / (1+math.exp(-(101.2355 + 16.7486*face_shape_index - 142.0216*face_jaw_index)))
    print("The probability of your face type belonging to 鹅蛋脸 or 瓜子脸 is '{}'".format(eg_probability))
    # 绘出轮廓
    landmarks_face = landmarks[4:13, :]
    landmarks_face_x = np.array(landmarks_face)[:,0]
    landmarks_face_y = np.array(landmarks_face)[:,1]
    face_line = np.poly1d(np.polyfit(landmarks_face_x, landmarks_face_y, 4))       #4存疑
    # for t in range(landmarks_face[0,0],landmarks_face[8,0],1):
    #     y_face = np.int(face_line(t))
    #     cv2.circle(cv_bgr_image, (t,y_face),2, color=(139, 0, 0))
    # 计算曲率半径
    face_line_derive = face_line.deriv()
    face_line_derive2 = face_line_derive.deriv()
    K = abs(face_line_derive2(int(face_point3[:,0]))) / (1 + face_line_derive(int(face_point3[:,0]))**2)**1.5
    r = 1 / K
    face_jaw_r = r / face_height
    if 13.8128 - 24.3904*face_shape_index + 16.4667*face_jaw_r > 0:
        guazi_probability = 0
        e_pro = 1 / (1+math.exp(-(13.8128 - 24.3904*face_shape_index + 16.4667*face_jaw_r))) * eg_probability
        print("The probability of your face type belonging to 鹅蛋脸 is '{}'".format(e_pro))
        print("The probability of your face type belonging to 瓜子脸 is '{}'".format((1 - e_pro / eg_probability) * eg_probability))
    else:
        edan_probability = 0
        g_pro = (1 - 1 / (1+math.exp(-(13.8128 - 24.3904*face_shape_index + 16.4667*face_jaw_r)))) * eg_probability
        print("The probability of your face type belonging to 瓜子脸 is '{}'".format(g_pro))
        print("The probability of your face type belonging to 鹅蛋脸 is '{}'".format((1 - g_pro / eg_probability) * eg_probability))
else:
    fang_probability = 1
    yuan_probability = 1
    fy_probability = 1 - 1 / (1+math.exp(-(101.2355 + 16.7486*face_shape_index - 142.0216*face_jaw_index)))
    print("The probability of your face type belonging to 方脸 or 圆脸 is '{}'".format(fy_probability))
    if -91.1403 + 54.8674 * face_shape_index + 82.285 * face_jaw_regression_index > 0:
        yuan_probability = 0
        f_pro = 1 / (1+math.exp(-(-91.1403 + 54.8674 * face_shape_index + 82.285 * face_jaw_regression_index))) * fy_probability
        print("The probability of your face type belonging to 方脸 is '{}'".format(f_pro))
        print("The probability of your eye type belonging to 圆眼 is '{}'".format(
            (1 - f_pro / fy_probability) * fy_probability))
    else:
        yuan_probability = 0
        y_pro = (1 - 1 / (1+math.exp(-(-91.1403 + 54.8674 * face_shape_index + 82.285 * face_jaw_regression_index)))) * fy_probability
        print("The probability of your face type belonging to 圆脸 is '{}'".format(y_pro))
        print("The probability of your face type belonging to 方脸 is '{}'".format((1 - y_pro / fy_probability) * fy_probability))





cv2.namedWindow("image", 1)
cv2.imshow("image", cv_bgr_image)
cv2.waitKey(0)




