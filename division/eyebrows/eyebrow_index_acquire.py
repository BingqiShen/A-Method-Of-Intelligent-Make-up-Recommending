import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import math
import os
import time
import shutil
import base64
import pandas as pd
import xlwt
#import XlsxWriter

#定义Canny边缘提取函数
def canny_detect(src):
    blurred = cv2.GaussianBlur(src, (3, 3), 0)  # 高斯模糊，去除干扰噪点
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


eye_distance_index = np.empty(shape=[0,2])
eye_distance_sum = 0



# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


file_path = 'eyebrow_jian'
file_lists = os.listdir(file_path)


eyebrow_index = np.empty(shape=[0,8])

i = 0
k = 0

for file_list in file_lists:
    # 读取图像文件
    img_rd = cv2.imread(file_path + '/' + file_list)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    rgb_img = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)   # opencv的颜色空间是BGR，需要转为RGB才能用在dlib中
    dets = detector(rgb_img, 1)

    # 检测到的人脸数量
    num_faces = len(dets)
    if num_faces != 1:
        print("Sorry, there were no faces found in '{}'".format(file_list))
        continue
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


                # # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
                # cv2.circle(cv_bgr_image, pos, 2, color=(139, 0, 0))
                # 利用 cv2.putText 写数字 1-68
                cv2.putText(cv_bgr_image, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

        # cv2.putText(cv_bgr_image, "faces: " + str(num_faces), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        # 没有检测到人脸
        cv2.putText(cv_bgr_image, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        continue

    ################################################################眉型参数###################################################################
    ########################################################瞳孔定位##########################################################
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
    eyeball_right_x = (int(eye_point4[:, 0]) + int(eye_point1[:, 0])) / 2
    eyeball_right_y = (min(int(eye_point2[:, 1]), int(eye_point3[:, 1])) + max(int(eye_point5[:, 1]),
                                                                               int(eye_point6[:, 1]))) / 2
    landmarks = np.vstack((landmarks, np.matrix([[eyeball_right_x, eyeball_right_y]])))
    eyeball_left_x = (int(eye_point7[:, 0]) + int(eye_point10[:, 0])) / 2
    eyeball_left_y = (min(int(eye_point8[:, 1]), int(eye_point9[:, 1])) + max(int(eye_point11[:, 1]),
                                                                              int(eye_point12[:, 1]))) / 2
    landmarks = np.vstack((landmarks, np.matrix([[eyeball_left_x, eyeball_left_y]])))
    # 定位右瞳孔
    pos1 = (int(landmarks[68][0, 0]), int(landmarks[68][0, 1]))
    pos2 = (int(landmarks[69][0, 0]), int(landmarks[6][0, 1]))


    eye_distance = (int(pos1[0] - pos2[0]) ** 2 + int(pos1[1] - pos2[1]) ** 2) ** 0.5
    # print(eye_distance)

    # 调整图像大小
    x, y = cv_bgr_image.shape[0:2]
    image_adjust = cv2.resize(cv_bgr_image,
                              (int(y / eye_distance * 327.3207535194756), int(x / eye_distance * 327.3207535194756)))


    ########################################################裁剪眉部图片##########################################################
    boudary_point1 = np.array(landmarks[17])
    boudary_point2 = np.array(landmarks[26])
    boudary_point3 = np.array(landmarks[37])
    boudary_point4 = np.array(landmarks[25])
    for j in range(3, 15, 1):
        eyebrow_image = image_adjust[int(boudary_point4[0,1])-7-j:int(boudary_point3[0,1] - j),int(boudary_point1[0,0])-1:int(boudary_point2[0,0])+2]
        ########################################################Canny特征提取##########################################################
        # 先进性闭运算，再进行特征提取
        eyebrow_image_close = close_demo(eyebrow_image)
        # cv2.imshow("eyebrow1",eyebrow_image_close)
        eyebrow_image_canny = canny_detect(eyebrow_image_close)
        # cv2.imshow("eyebrow2",eyebrow_image_canny)
        # cv2.waitKey(0)

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
                    try:
                        cnt = contours[3]  # 取第四条轮廓
                        # 计算眉毛区域的面积 + 周长
                        area = cv2.contourArea(cnt)
                        perimeter = cv2.arcLength(cnt, True)
                        if area > 500:
                            break
                    except:
                        continue
    # print(j)
    # if area > 500:
    #     k = k + 1
    #     print("The eyebrows found in '{}' are ok".format(file_list))



    # 计算眉毛宽度
    eyebrow_width = int(np.array(landmarks[21])[:, 0] - np.array(landmarks[17])[:, 0])
    # 计算眉毛斜度
    eyebrow_slope = math.atan((0.5 * (float(np.array(landmarks[21])[:, 1] - np.array(landmarks[17])[:, 1]) + float(np.array(landmarks[22])[:, 1] - np.array(landmarks[26])[:, 1]))) / eyebrow_width)
    # 计算眉峰
    eyebrow_peak = 0.5 * (int(np.array(landmarks[18])[:, 1] - np.array(landmarks[19])[:, 1]) + int(np.array(landmarks[25])[:, 1] - np.array(landmarks[24])[:, 1]))
    # 计算眉峰与眉尖正切
    if np.array(landmarks[18])[:, 1] < np.array(landmarks[19])[:, 1]:
        eyebrow_peak_point = landmarks[18]
    else:
        eyebrow_peak_point = landmarks[19]
    eyebrow_peak_tan = math.atan(float(np.array(landmarks[21])[:, 1] - eyebrow_peak_point[:,1]) / float(np.array(landmarks[21])[:, 0] - eyebrow_peak_point[:,0]))

    # 左眉拟合
    landmarks_eyebrow_left = landmarks[17:22, :]
    landmarks_eyebrow_left_x = np.array(landmarks_eyebrow_left)[:, 0]
    landmarks_eyebrow_left_y = np.array(landmarks_eyebrow_left)[:, 1]
    eyebrow_line_left = np.poly1d(np.polyfit(landmarks_eyebrow_left_x, landmarks_eyebrow_left_y, 3))  # 5存疑

    # 右眉拟合
    landmarks_eyebrow_right = landmarks[22:27, :]
    landmarks_eyebrow_right_x = np.array(landmarks_eyebrow_right)[:, 0]
    landmarks_eyebrow_right_y = np.array(landmarks_eyebrow_right)[:, 1]
    eyebrow_line_right = np.poly1d(np.polyfit(landmarks_eyebrow_right_x, landmarks_eyebrow_right_y, 3))  # 5存疑

    # 计算曲率半径
    eyebrow_line_left_derive = eyebrow_line_left.deriv()
    eyebrow_line_left_derive2 = eyebrow_line_left_derive.deriv()
    K_left = abs(eyebrow_line_left_derive2(int(np.array(landmarks[19])[:, 0]))) / (
                1 + eyebrow_line_left_derive(int(np.array(landmarks[19])[:, 0])) ** 2) ** 1.5
    eyebrow_line_right_derive = eyebrow_line_right.deriv()
    eyebrow_line_right_derive2 = eyebrow_line_right_derive.deriv()
    K_right = abs(eyebrow_line_right_derive2(int(np.array(landmarks[24])[:, 0]))) / (
                1 + eyebrow_line_right_derive(int(np.array(landmarks[24])[:, 0])) ** 2) ** 1.5

    K = 0.5 * (K_left + K_right)
    r = 1 / K
    eyebrow_r = r / eyebrow_width

    # if eyebrow_r < 0.75:
    #     print("The eyebrows found in '{}' aren't ok".format(file_list))
    #     print(eyebrow_r)

    eyebrow_index = np.append(eyebrow_index, [[area, perimeter, eyebrow_width, eyebrow_r, file_list,eyebrow_slope,eyebrow_peak,eyebrow_peak_tan]], axis=0)

    i = i + 1
    print(i)


print(len(eyebrow_index))
####################################################将眉毛数据导入至Excel中#####################################################

# workbook = xlwt.Workbook(encoding = 'ascii')        # 创建一个工作表
# worksheet = workbook.add_sheet('sheet1')            # 创建一个工作表对象
# for i in range(len(eyebrow_index)):
#     worksheet.write(i,0,eyebrow_index[i,0])
#     worksheet.write(i,1,eyebrow_index[i,1])
#     worksheet.write(i,2,eyebrow_index[i,2])
#     worksheet.write(i,3,eyebrow_index[i,3])
#     worksheet.write(i,4,eyebrow_index[i,4])
# workbook.save('eyebrow_biao_slope_index.xls')        # 关闭Excel文件

workbook = xlwt.Workbook(encoding = 'ascii')        # 创建一个工作表
worksheet = workbook.add_sheet('sheet1')            # 创建一个工作表对象
for i in range(len(eyebrow_index)):
    worksheet.write(i,0,eyebrow_index[i,7])
workbook.save('eyebrow_jian_peak_index.xls')        # 关闭Excel文件
##############################################################################################################################




