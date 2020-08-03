# -*- coding:UTF-8 -*-
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
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import detect_Baidu
import sys

face_type = 0
eye_type = 0
eyebrow_type = 0
nose_type = 0
img_path = "2.jpg"

def landmark_adjust(img_path):
    # Dlib 检测器和预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    img_rd = cv2.imread(img_path)
    img_rd = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    rgb_img = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)   # opencv的颜色空间是BGR，需要转为RGB才能用在dlib中
    dets = detector(rgb_img, 1)

    # 检测到的人脸数量
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(img_path))
        exit()

    # 识别人脸特征点，并保存下来
    faces = dlib.full_object_detections()
    for det in dets:
        faces.append(predictor(rgb_img, det))

    # 待会要写的字体
    global font
    font = cv2.FONT_HERSHEY_SIMPLEX

    global cv_bgr_image
    ################################################################人脸对齐###################################################################
    images = dlib.get_face_chips(rgb_img, faces, size=500)
    # 显示计数，按照这个计数创建窗口
    image_cnt = 0
    # 显示对齐结果
    for image in images:
        image_cnt += 1
        cv_rgb_image = np.array(image).astype(np.uint8)  # 先转换为numpy数组
        cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)  # opencv下颜色空间为bgr，所以从rgb转换为bgr
        # cv2.imshow('%s' % (image_cnt), cv_bgr_image)
        # cv2.waitKey(0)

    ################################################################关键点确定###################################################################

    cv_bgr_image_grey = cv2.cvtColor(cv_bgr_image, cv2.COLOR_RGB2GRAY)
    dets_adjust = detector(cv_bgr_image_grey, 0)

    if num_faces == 1:
        # 检测到一张人脸

        # 取特征点坐标
        landmarks = np.matrix([[p.x, p.y] for p in predictor(cv_bgr_image, dets_adjust[0]).parts()])
        cv2.rectangle(cv_bgr_image, (dlib.rectangle().left(), dlib.rectangle().top()), (dlib.rectangle().right(), dlib.rectangle().bottom()), (255, 255, 255))
        shape=predictor(cv_bgr_image,dlib.rectangle())

        # cv2.putText(cv_bgr_image, "faces: " + str(num_faces), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        # 没有检测到人脸
        cv2.putText(cv_bgr_image, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

    return landmarks

################################################################脸型分类###################################################################
def face_division(landmarks):
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
            face_type = 1
        else:
            edan_probability = 0
            g_pro = (1 - 1 / (1+math.exp(-(13.8128 - 24.3904*face_shape_index + 16.4667*face_jaw_r)))) * eg_probability
            print("The probability of your face type belonging to 瓜子脸 is '{}'".format(g_pro))
            print("The probability of your face type belonging to 鹅蛋脸 is '{}'".format((1 - g_pro / eg_probability) * eg_probability))
            face_type = 2
    else:
        fang_probability = 1
        yuan_probability = 1
        fy_probability = 1 - 1 / (1+math.exp(-(101.2355 + 16.7486*face_shape_index - 142.0216*face_jaw_index)))
        print("The probability of your face type belonging to 方脸 or 圆脸 is '{}'".format(fy_probability))
        if -91.1403 + 54.8674 * face_shape_index + 82.285 * face_jaw_regression_index > 0:
            yuan_probability = 0
            f_pro = 1 / (1+math.exp(-(-91.1403 + 54.8674 * face_shape_index + 82.285 * face_jaw_regression_index))) * fy_probability
            print("The probability of your face type belonging to 方脸 is '{}'".format(f_pro))
            print("The probability of your eye type belonging to 圆脸 is '{}'".format(
                (1 - f_pro / fy_probability) * fy_probability))
            face_type = 3
        else:
            yuan_probability = 0
            y_pro = (1 - 1 / (1+math.exp(-(-91.1403 + 54.8674 * face_shape_index + 82.285 * face_jaw_regression_index)))) * fy_probability
            print("The probability of your face type belonging to 圆脸 is '{}'".format(y_pro))
            print("The probability of your face type belonging to 方脸 is '{}'".format((1 - y_pro / fy_probability) * fy_probability))
            face_type = 4
    return face_type

################################################################眼型分类###################################################################
def eyes_division(landmarks):
    eye_point1 = np.array(landmarks[36])
    eye_point2 = np.array(landmarks[37])
    eye_point3 = np.array(landmarks[38])
    eye_point4 = np.array(landmarks[39])
    eye_point5 = np.array(landmarks[40])
    eye_point6 = np.array(landmarks[41])
    eye_point7 = np.array(landmarks[42])
    eye_point8 = np.array(landmarks[43])
    eye_point9 = np.array(landmarks[44])
    eye_point10 = np.array(landmarks[45])
    eye_point11 = np.array(landmarks[46])
    eye_point12 = np.array(landmarks[47])
    eyeball_right_x = (int(eye_point4[:,0]) + int(eye_point1[:,0])) / 2
    eyeball_right_y = (min(int(eye_point2[:,1]),int(eye_point3[:,1])) + max(int(eye_point5[:,1]),int(eye_point6[:,1]))) / 2
    landmarks = np.vstack((landmarks,np.matrix([[eyeball_right_x,eyeball_right_y]])))
    eyeball_left_x = (int(eye_point7[:,0]) + int(eye_point10[:,0])) / 2
    eyeball_left_y = (min(int(eye_point8[:,1]),int(eye_point9[:,1])) + max(int(eye_point11[:,1]),int(eye_point12[:,1]))) / 2
    landmarks = np.vstack((landmarks,np.matrix([[eyeball_left_x,eyeball_left_y]])))
    #定位右瞳孔
    pos1 = (int(landmarks[68][0,0]),int(landmarks[68][0,1]))
    # cv2.circle(cv_bgr_image, pos1, 2, color=(139, 0, 0))
    # cv2.putText(cv_bgr_image, str(69), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)
    #定位左瞳孔
    pos2 = (int(landmarks[69][0,0]),int(landmarks[6][0,1]))
    # cv2.circle(cv_bgr_image, pos2, 2, color=(139, 0, 0))
    # cv2.putText(cv_bgr_image, str(70), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

    #########################################################获取眼部参数#########################################################
    eye_length = (int(eye_point4[:,0]) - int(eye_point1[:,0]) + int(eye_point10[:,0]) - int(eye_point7[:,0]))/2
    eye_height = (-min(int(eye_point2[:,1]),int(eye_point3[:,1])) + max(int(eye_point5[:,1]),int(eye_point6[:,1])) - min(int(eye_point8[:,1]),int(eye_point9[:,1])) + max(int(eye_point11[:,1]),int(eye_point12[:,1])))/2
    #表示眼睛椭圆程度
    eye_d = eye_height/eye_length
    #表示眼睛倾斜角度
    eye_tan = ((int(eye_point4[:,1]) - int(eye_point1[:,1])) / (int(eye_point4[:,0]) - int(eye_point1[:,0])) + (int(eye_point7[:,1]) - int(eye_point10[:,1])) / (int(eye_point10[:,0]) - int(eye_point7[:,0]))) / 2



    #以下数据来自matlab数据分析
    if -22.6456 + 66.9587 * eye_d - 12.6067 * eye_tan > 0:   #可能是杏眼或者桃花眼
        xt_probability = 1 / (1 + math.exp(-(-22.6456 + 66.9587 * eye_d - 12.6067 * eye_tan)))
        print("The probability of your eye type belonging to 杏眼 or 桃花眼 is '{}'".format(xt_probability))
        # 对上眼线拟合
        landmarks_eye_left_up = landmarks[36:40, :]
        landmarks_eye_right_up = landmarks[42:46, :]
        landmarks_eye_left_up_x = np.array(landmarks_eye_left_up)[:, 0]
        landmarks_eye_left_up_y = np.array(landmarks_eye_left_up)[:, 1]
        landmarks_eye_right_up_x = np.array(landmarks_eye_right_up)[:, 0]
        landmarks_eye_right_up_y = np.array(landmarks_eye_right_up)[:, 1]

        eye_left_up_line = np.poly1d(np.polyfit(landmarks_eye_left_up_x, landmarks_eye_left_up_y, 3))  # 3存疑
        eye_right_up_line = np.poly1d(np.polyfit(landmarks_eye_right_up_x, landmarks_eye_right_up_y, 3))  # 3存疑

        # 对下眼线拟合
        landmarks_eye_left_down = np.vstack((landmarks[39:42, :], landmarks[36, :]))
        landmarks_eye_right_down = np.vstack((landmarks[45:48, :], landmarks[42, :]))
        landmarks_eye_left_down_x = np.array(landmarks_eye_left_down)[:, 0]
        landmarks_eye_left_down_y = np.array(landmarks_eye_left_down)[:, 1]
        landmarks_eye_right_down_x = np.array(landmarks_eye_right_down)[:, 0]
        landmarks_eye_right_down_y = np.array(landmarks_eye_right_down)[:, 1]

        eye_left_down_line = np.poly1d(np.polyfit(landmarks_eye_left_down_x, landmarks_eye_left_down_y, 3))  # 3存疑
        eye_right_down_line = np.poly1d(np.polyfit(landmarks_eye_right_down_x, landmarks_eye_right_down_y, 3))  # 3存疑

        # 计算眼角斜率
        eye_left_up_line_derive = eye_left_up_line.deriv()
        eye_left_down_line_derive = eye_left_down_line.deriv()
        eye_left_innercorner_angle = math.atan(
            eye_left_up_line_derive(int(eye_point4[:, 0])) - eye_left_down_line_derive(int(eye_point4[:, 0])))
        eye_left_outercorner_angle = math.atan(
            eye_left_down_line_derive(int(eye_point1[:, 0])) - eye_left_up_line_derive(int(eye_point1[:, 0])))

        eye_right_up_line_derive = eye_right_up_line.deriv()
        eye_right_down_line_derive = eye_right_down_line.deriv()
        eye_right_innercorner_angle = math.atan(
            eye_right_down_line_derive(int(eye_point7[:, 0])) - eye_right_up_line_derive(int(eye_point7[:, 0])))
        eye_right_outercorner_angle = math.atan(
            eye_right_up_line_derive(int(eye_point10[:, 0])) - eye_right_down_line_derive(int(eye_point10[:, 0])))

        eye_innercorner_angle = (eye_left_innercorner_angle + eye_right_innercorner_angle) / 2
        eye_outercorner_angle = (eye_left_outercorner_angle + eye_right_outercorner_angle) / 2

        if -76.3111 + 35.1321 * eye_innercorner_angle +36.8774 * eye_outercorner_angle +2.2487 * eye_d > 0:
            x_pro = 1 / (1+math.exp(-(-76.3111 + 35.1321 * eye_innercorner_angle +36.8774 * eye_outercorner_angle +2.2487 * eye_d))) * xt_probability
            print("The probability of your eye type belonging to 杏眼 is '{}'".format(x_pro))
            print("The probability of your eye type belonging to 桃花眼 is '{}'".format(
                (1 - x_pro / xt_probability) * xt_probability))
            eye_type = 1
        else:
            t_pro = (1 - 1 / (1+math.exp(-(-76.3111 + 35.1321 * eye_innercorner_angle +36.8774 * eye_outercorner_angle +2.2487 * eye_d)))) * xt_probability
            print("The probability of your eye type belonging to 桃花眼 is '{}'".format(t_pro))
            print("The probability of your eye type belonging to 杏眼 is '{}'".format(
                (1 - t_pro / xt_probability) * xt_probability))
            eye_type = 2

    else:                                                    #可能是丹凤眼或柳叶眼
        ld_probability = 1 - 1 / (1 + math.exp(-(-22.6456 + 66.9587 * eye_d - 12.6067 * eye_tan)))
        print("The probability of your eye type belonging to 柳叶眼 or 丹凤眼 is '{}'".format(ld_probability))
        # 计算上部曲线曲率半径
           # 绘出轮廓
        landmarks_eye_left = landmarks[36:40, :]
        landmarks_eye_right = landmarks[42:46, :]
        landmarks_eye_left_x = np.array(landmarks_eye_left)[:, 0]
        landmarks_eye_left_y = np.array(landmarks_eye_left)[:, 1]
        landmarks_eye_right_x = np.array(landmarks_eye_right)[:, 0]
        landmarks_eye_right_y = np.array(landmarks_eye_right)[:, 1]

        eye_left_line = np.poly1d(np.polyfit(landmarks_eye_left_x, landmarks_eye_left_y, 3))  # 3存疑
        eye_right_line = np.poly1d(np.polyfit(landmarks_eye_right_x, landmarks_eye_right_y, 3))  # 3存疑
           # 计算曲率半径
        eye_left_line_derive = eye_left_line.deriv()
        eye_left_line_derive2 = eye_left_line_derive.deriv()
        K_left = abs(eye_left_line_derive2(int(eye_point2[:, 0]))) / (
                    1 + eye_left_line_derive(int(eye_point2[:, 0])) ** 2) ** 1.5
        r_left = 1 / K_left

        eye_right_line_derive = eye_right_line.deriv()
        eye_right_line_derive2 = eye_right_line_derive.deriv()
        K_right = abs(eye_right_line_derive2(int(eye_point9[:, 0]))) / (
                1 + eye_right_line_derive(int(eye_point9[:, 0])) ** 2) ** 1.5
        r_right = 1 / K_right

        r = (r_left + r_right) / 2
        eye_r = r / eye_length

        if 21.9791 - 10.8691 * eye_tan - 28.2903 * eye_r > 0:
            l_pro = 1 / (1+math.exp(-(21.9791 - 10.8691 * eye_tan - 28.2903 * eye_r))) * ld_probability
            print("The probability of your eye type belonging to 柳叶眼 is '{}'".format(l_pro))
            print("The probability of your eye type belonging to 丹凤眼 is '{}'".format(
                (1 - l_pro / ld_probability) * ld_probability))
            eye_type = 3
        else:
            d_pro = (1 - 1 / (1+math.exp(-(21.9791 - 10.8691 * eye_tan - 28.2903 * eye_r)))) * ld_probability
            print("The probability of your eye type belonging to 丹凤眼 is '{}'".format(d_pro))
            print("The probability of your eye type belonging to 柳叶眼 is '{}'".format((1 - d_pro / ld_probability) * ld_probability))
            eye_type = 4
    return eye_type, pos1, pos2

################################################################眉型分类###################################################################
def eyebrows_division(landmarks, pos1, pos2):
###################################################训练SVM模型############################################################
    path = 'eyebrow_data.xls'  # 数据文件路径
    data = pd.read_excel(path, dtype=float, delimiter=',')
    data = np.array(data)
    eyebrow_type = 0

    # 引入并训练数据
    x, y = np.split(data, (7,), axis=1)
    # Feature Scaling
    scaler = preprocessing.StandardScaler().fit(x)
    x = preprocessing.scale(x)
    #
    # train_data,test_data,train_label,test_label = train_test_split(x, y, random_state=0, train_size=0.7,test_size=0.3)
    # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    # clf.fit(x, y.ravel())
    #
    # # 保存model
    # joblib.dump(clf,'clf_eyebrow.pkl')

    # 定义一个存放眉毛特征值的矩阵
    eyebrow_index = np.empty(shape=[0,7])

    #定义Canny边缘提取函数
    def canny_detect(src):
        blurred = cv2.GaussianBlur(src, (3, 3), 0)  # 高斯模糊，去除干扰噪点
        # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
        # X gradient
        gradx = cv2.Sobel(blurred, cv2.CV_16SC1, 1, 0)
        # Y gradient
        grady = cv2.Sobel(blurred, cv2.CV_16SC1, 0, 1)
        # edge
        edge_output = cv2.Canny(gradx, grady, 50, 150)  # 利用x，y 轴梯度对图像进行边缘提取
        return edge_output

    # 定义闭运算
    def close_demo(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    # 由眼型参数得
    eye_distance = (int(pos1[0]-pos2[0])**2 + int(pos1[1]-pos2[1])**2)**0.5
    #print(eye_distance)

    # 调整图像大小
    x, y = cv_bgr_image.shape[0:2]
    image_adjust = cv2.resize(cv_bgr_image,(int(y /eye_distance*327.3207535194756),int(x /eye_distance*327.3207535194756)))

    # cv2.imshow('image', image_adjust)
    # cv2.waitKey(0)

    ########################################################裁剪眉部图片##########################################################
    boudary_point1 = np.array(landmarks[17])
    boudary_point2 = np.array(landmarks[26])
    boudary_point3 = np.array(landmarks[37])
    boudary_point4 = np.array(landmarks[25])
    for j in range(3,20,1):
        eyebrow_image = image_adjust[int(boudary_point4[0,1])-7-j:int(boudary_point3[0,1] - j+5),int(boudary_point1[0,0])-6:int(boudary_point2[0,0])+15]
        ########################################################Canny特征提取##########################################################
        #先进性闭运算，再进行特征提取
        eyebrow_image_close = close_demo(eyebrow_image)
        # cv2.imshow("eyebrow1",eyebrow_image_close)
        eyebrow_image_canny = canny_detect(eyebrow_image_close)
        cv2.imshow("eyebrow2",eyebrow_image_canny)
        cv2.waitKey(0)

        ret, thresh = cv2.threshold(eyebrow_image_canny, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 得到轮廓信息
        cnt = contours[0]  # 取第一条轮廓
        # 计算眉毛区域的面积 + 周长
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        # 大于1000表示截到了纯眉毛图片
        if area > 500:
            break
        else:
            try:
                cnt = contours[1]  # 取第二条轮廓
                # 计算眉毛区域的面积 + 周长
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if area > 500:
                    break
                else:
                    try:
                        cnt = contours[2]  # 取第三条轮廓
                        # 计算眉毛区域的面积 + 周长
                        area = cv2.contourArea(cnt)
                        perimeter = cv2.arcLength(cnt, True)
                        if area > 500:
                            break
                    except:
                        continue
            except:
                continue

    ################################################ 眉型特征值 ###################################################
    # print('面积：',area)
    # print('周长：',perimeter)
    #计算眉毛宽度
    eyebrow_width = int(np.array(landmarks[21])[:,0] - np.array(landmarks[17])[:,0])
    # print('眉毛宽度：',eyebrow_width)
    #
    # 计算眉毛斜度
    eyebrow_slope = math.atan((0.5 * (float(np.array(landmarks[21])[:, 1] - np.array(landmarks[17])[:, 1]) + float(np.array(landmarks[22])[:, 1] - np.array(landmarks[26])[:, 1]))) / eyebrow_width)
    # print('眉毛斜度：', eyebrow_slope)
    # 计算眉峰
    eyebrow_peak = 0.5 * (int(np.array(landmarks[18])[:, 1] - np.array(landmarks[19])[:, 1]) + int(np.array(landmarks[25])[:, 1] - np.array(landmarks[24])[:, 1]))
    # print('眉峰：', eyebrow_peak)
    # 计算眉峰与眉尖正切
    if np.array(landmarks[18])[:, 1] < np.array(landmarks[19])[:, 1]:
        eyebrow_peak_point = landmarks[18]
    else:
        eyebrow_peak_point = landmarks[19]
    eyebrow_peak_tan = math.atan(float(np.array(landmarks[21])[:, 1] - eyebrow_peak_point[:,1]) / float(np.array(landmarks[21])[:, 0] - eyebrow_peak_point[:,0]))
    # print('眉峰与眉尖正切：', eyebrow_peak_tan)

    #左眉拟合
    landmarks_eyebrow_left = landmarks[17:22, :]
    landmarks_eyebrow_left_x = np.array(landmarks_eyebrow_left)[:,0]
    landmarks_eyebrow_left_y = np.array(landmarks_eyebrow_left)[:,1]
    eyebrow_line_left = np.poly1d(np.polyfit(landmarks_eyebrow_left_x, landmarks_eyebrow_left_y, 3))       #5存疑

    #右眉拟合
    landmarks_eyebrow_right = landmarks[22:27, :]
    landmarks_eyebrow_right_x = np.array(landmarks_eyebrow_right)[:,0]
    landmarks_eyebrow_right_y = np.array(landmarks_eyebrow_right)[:,1]
    eyebrow_line_right = np.poly1d(np.polyfit(landmarks_eyebrow_right_x, landmarks_eyebrow_right_y, 3))       #5存疑

    # 计算曲率半径
    eyebrow_line_left_derive = eyebrow_line_left.deriv()
    eyebrow_line_left_derive2 = eyebrow_line_left_derive.deriv()
    K_left = abs(eyebrow_line_left_derive2(int(np.array(landmarks[19])[:, 0]))) / (1 + eyebrow_line_left_derive(int(np.array(landmarks[19])[:, 0])) ** 2) ** 1.5
    eyebrow_line_right_derive = eyebrow_line_right.deriv()
    eyebrow_line_right_derive2 = eyebrow_line_right_derive.deriv()
    K_right = abs(eyebrow_line_right_derive2(int(np.array(landmarks[24])[:, 0]))) / (1 + eyebrow_line_right_derive(int(np.array(landmarks[24])[:, 0])) ** 2) ** 1.5

    K = 0.5 * (K_left + K_right)
    r = 1 / K
    eyebrow_r = r / eyebrow_width
    # print('眉毛曲率半径：',eyebrow_r)


    if eyebrow_r > 1:
        eyebrow_type == 4
        print('Your eyebrows type belongs to 剑眉')
        return eyebrow_type


    if area < 500:
        print('眉毛有遮挡，无法识别！')
        eyebrow_type = 0
        return eyebrow_type

    # 眉毛特征值矩阵生成
    eyebrow_index = np.append(eyebrow_index, [[area, perimeter, eyebrow_width, eyebrow_r, eyebrow_peak, eyebrow_slope, eyebrow_peak_tan]], axis=0)
    eyebrow_index_standard = scaler.transform(eyebrow_index)

    # 导入模型
    clf_eyebrow = joblib.load('clf_eyebrow.pkl')
    outcome = int(clf_eyebrow.predict(eyebrow_index_standard))
    eyebrow_type = outcome
    if outcome == 1:
        print('Your eyebrows type belongs to 标准眉')
    elif outcome  == 2:
        print('Your eyebrows type belongs to 一字眉')
    elif outcome == 3:
        print('Your eyebrows type belongs to 柳叶眉')
    elif outcome == 4:
        print('Your eyebrows type belongs to 剑眉')

    return eyebrow_type


################################################################鼻型分类#####################################################################
def nose_division(landmarks):
    path = 'nose_data.xlsx'  # 数据文件路径
    data = pd.read_excel(path, dtype=float, delimiter=',')
    data = np.array(data)

    # 引入并训练数据
    x, y = np.split(data, (4,), axis=1)
    # Feature Scaling
    scaler = preprocessing.StandardScaler().fit(x)
    x = preprocessing.scale(x)

    # train_data,test_data,train_label,test_label = train_test_split(x, y, random_state=0, train_size=0.7,test_size=0.3)
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    # clf.fit(x, y.ravel())
    #
    # # 保存model
    # joblib.dump(clf,'clf_nose.pkl')

    nose_index = np.empty(shape=[0,4])

    nose_point1 = landmarks[27]
    nose_point2 = landmarks[28]
    nose_point3 = landmarks[29]
    nose_point4 = landmarks[30]
    nose_point5 = landmarks[31]
    nose_point6 = landmarks[32]
    nose_point7 = landmarks[33]
    nose_point8 = landmarks[34]
    nose_point9 = landmarks[35]

    face_height = np.linalg.norm(nose_point1 - landmarks[8])
    nose_height = int(nose_point7[:, 1] - nose_point1[:, 1])
    nose_width = int(nose_point9[:, 0] - nose_point5[:, 0])
    nose_length = int(nose_point4[:, 1] - nose_point1[:, 1])
    # 用来判断水滴鼻
    nose_arctan = (math.atan((int(nose_point5[:, 1]) - int(nose_point4[:, 1])) / (
                int(nose_point4[:, 0]) - int(nose_point5[:, 0]))) + math.atan(
        (int(nose_point9[:, 1]) - int(nose_point4[:, 1])) / (int(nose_point9[:, 0]) - int(nose_point4[:, 0])))) / 2
    # 用来判断希腊鼻和狮子鼻
    nose_wh = nose_width / nose_height
    nose_lh = nose_length / nose_height
    # 用来判断短鼻
    nose_relative_height = nose_length / face_height

    nose_index = np.append(nose_index, [[nose_wh, nose_lh, nose_relative_height,nose_arctan]], axis=0)
    nose_index_standard = scaler.transform(nose_index)
    # 导入模型
    clf_nose = joblib.load('clf_nose.pkl')
    outcome = int(clf_nose.predict(nose_index_standard))
    nose_type = outcome
    if outcome == 1:
        print('Your nose type belongs to 狮子鼻')
    elif outcome  == 2:
        print('Your nose type belongs to 希腊鼻/直鼻')
    elif outcome == 3:
        print('Your nose type belongs to 水滴鼻')
    elif outcome == 4:
        print('Your nose type belongs to 短鼻')

    return nose_type

###########################################################关键点标定##########################################################
def draw_landmarks(landmarks):
    for idx, point in enumerate(landmarks):
        # 68 点的坐标
        pos = (int(point[0, 0]), int(point[0, 1]))

        #利用 cv2.circle 给每个特征点画一个圈，共 68 个
        cv2.circle(cv_bgr_image, pos, 2, color=(139, 0, 0))
        #利用 cv2.putText 写数字 1-68
        cv2.putText(cv_bgr_image, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("image", cv_bgr_image)
    cv2.waitKey(0)

# 主函数
if __name__ == '__main__':
    a = detect_Baidu.face_information(img_path)
    try:
        yaw, pitch, roll = detect_Baidu.face_angle(a)
        blur, illumination = detect_Baidu.face_blur_illumination(a)
        if abs(yaw) < 15 and abs(pitch) < 20 and blur < 0.7 and illumination > 80:
            landmarks = landmark_adjust(img_path)
            face_type = face_division(landmarks)
            eye_type, pos1, pos2 = eyes_division(landmarks)
            eyebrow_type = eyebrows_division(landmarks, pos1, pos2)
            nose_type = nose_division(landmarks)
            draw_landmarks(landmarks)
        elif abs(yaw) > 15:
            print('请勿侧脸！')
        elif abs(pitch) > 20:
            print('请勿低头或抬头！')
        elif illumination < 80:
            print('请在光照充足环境下拍照！')
        elif blur > 7:
            print('请上传清晰照片！')
    except:
        print('未识别到人脸')

