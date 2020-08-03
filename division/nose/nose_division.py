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


# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 读取图像文件
img_rd = cv2.imread("nose_xi/xi3.jpg")
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

if num_faces != 0:
    # 检测到人脸

    for i in range(num_faces):
        # 取特征点坐标
        landmarks = np.matrix([[p.x, p.y] for p in predictor(cv_bgr_image, dets_adjust[i]).parts()])
        #landmarks_face = landmarks[0:17, :]
        cv2.rectangle(cv_bgr_image, (dlib.rectangle().left(), dlib.rectangle().top()), (dlib.rectangle().right(), dlib.rectangle().bottom()), (255, 255, 255))
        shape=predictor(cv_bgr_image,dlib.rectangle())
        for idx, point in enumerate(landmarks):
            # 68 点的坐标
            pos = (point[0, 0], point[0, 1])



            # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
            cv2.circle(cv_bgr_image, pos, 2, color=(139, 0, 0))
            # 利用 cv2.putText 写数字 1-68
            cv2.putText(cv_bgr_image, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(cv_bgr_image, "faces: " + str(num_faces), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
else:
    # 没有检测到人脸
    cv2.putText(cv_bgr_image, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)


########################################################鼻子参数##########################################################
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




path = 'nose_data.xlsx'  # 数据文件路径
data = pd.read_excel(path, dtype=float, delimiter=',')
data = np.array(data)

# 引入并训练数据
x, y = np.split(data, (4,), axis=1)
# Feature Scaling
scaler = preprocessing.StandardScaler().fit(x)
x = preprocessing.scale(x)

train_data,test_data,train_label,test_label = train_test_split(x, y, random_state=0, train_size=0.7,test_size=0.3)
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x, y.ravel())

# 保存model
joblib.dump(clf,'clf_nose.pkl')

nose_index_standard = scaler.transform(nose_index)
# 导入模型
clf_eyebrow = joblib.load('clf_nose.pkl')
outcome = int(clf_eyebrow.predict(nose_index_standard))
if outcome == 1:
    print('Your nose type belongs to 狮子鼻')
elif outcome  == 2:
    print('Your nose type belongs to 希腊鼻/直鼻')
elif outcome == 3:
    print('Your nose type belongs to 水滴鼻')
elif outcome == 4:
    print('Your nose type belongs to 短鼻')

cv2.namedWindow("image", 1)
cv2.imshow("image", cv_bgr_image)
cv2.waitKey(0)