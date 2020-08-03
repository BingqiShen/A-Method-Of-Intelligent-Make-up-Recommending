import numpy as np  # 数据处理的库 numpy
import dlib         # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv
import math


# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 读取图像文件
img_rd = cv2.imread("eye_danfeng/danfeng13.jpg")
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
eyeball_right_x = (int(eye_point4[:,0]) + int(eye_point1[:,0])) / 2
eyeball_right_y = (min(int(eye_point2[:,1]),int(eye_point3[:,1])) + max(int(eye_point5[:,1]),int(eye_point6[:,1]))) / 2
landmarks = np.vstack((landmarks,np.matrix([[eyeball_right_x,eyeball_right_y]])))
eyeball_left_x = (int(eye_point7[:,0]) + int(eye_point10[:,0])) / 2
eyeball_left_y = (min(int(eye_point8[:,1]),int(eye_point9[:,1])) + max(int(eye_point11[:,1]),int(eye_point12[:,1]))) / 2
landmarks = np.vstack((landmarks,np.matrix([[eyeball_left_x,eyeball_left_y]])))
#定位右瞳孔
pos1 = (int(landmarks[68][0,0]),int(landmarks[68][0,1]))
cv2.circle(cv_bgr_image, pos1, 2, color=(139, 0, 0))
cv2.putText(cv_bgr_image, str(69), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)
#定位左瞳孔
pos2 = (int(landmarks[69][0,0]),int(landmarks[6][0,1]))
cv2.circle(cv_bgr_image, pos2, 2, color=(139, 0, 0))
cv2.putText(cv_bgr_image, str(70), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)

########################################################裁剪眼部图片##########################################################
# boudary_point1 = np.array(landmarks[17])
# boudary_point2 = np.array(landmarks[26])
# boudary_point3 = np.array(landmarks[28])
# eye_image = cv_bgr_image[int(boudary_point1[0,1]):int(boudary_point3[0,1]),int(boudary_point1[0,0]):int(boudary_point2[0,0])]
# cv2.namedWindow("eye_image", 1)
# cv2.imshow("eye_image", eye_image)
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
    print("The probability of your eye type belonging to xing or taohua is '{}'".format(xt_probability))
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
        print("The probability of your eye type belonging to xing is '{}'".format(x_pro))
        print("The probability of your eye type belonging to taohua is '{}'".format(
            (1 - x_pro / xt_probability) * xt_probability))
    else:
        t_pro = (1 - 1 / (1+math.exp(-(-76.3111 + 35.1321 * eye_innercorner_angle +36.8774 * eye_outercorner_angle +2.2487 * eye_d)))) * xt_probability
        print("The probability of your eye type belonging to taohua is '{}'".format(t_pro))
        print("The probability of your eye type belonging to xing is '{}'".format(
            (1 - t_pro / xt_probability) * xt_probability))

else:                                                    #可能是丹凤眼或柳叶眼
    ld_probability = 1 - 1 / (1 + math.exp(-(-22.6456 + 66.9587 * eye_d - 12.6067 * eye_tan)))
    print("The probability of your eye type belonging to liuye or danfeng is '{}'".format(ld_probability))
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
        print("The probability of your eye type belonging to liuye is '{}'".format(l_pro))
        print("The probability of your eye type belonging to danfeng is '{}'".format(
            (1 - l_pro / ld_probability) * ld_probability))
    else:
        d_pro = (1 - 1 / (1+math.exp(-(21.9791 - 10.8691 * eye_tan - 28.2903 * eye_r)))) * ld_probability
        print("The probability of your eye type belonging to danfeng is '{}'".format(d_pro))
        print("The probability of your eye type belonging to liuye is '{}'".format((1 - d_pro / ld_probability) * ld_probability))


cv2.namedWindow("image", 1)
cv2.imshow("image", cv_bgr_image)
cv2.waitKey(0)



# 灰度图
# eye_image_grey = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
# Max = max(max(eye_image_grey))
# print(Max)
# #cv2.imshow("gray image", eye_image_grey)
# # 二值化图
# ret,binary = cv2.threshold(eye_image_grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# #cv2.imshow("binary image", binary)
# element1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3),(-1,-1))
# tmp = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,element1,None,(-1,-1),1)
# #cv2.imshow("tmp image", tmp)
#
# element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(16,16),(-1,-1))
# dst = cv2.morphologyEx(tmp,cv2.MORPH_OPEN,element2)
#cv2.imshow("eye image", dst)

# cloneImage, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, None, None, (0, 0))
# for i, contour in enumerate(contours):
#     print ("find ")
#     cv2.drawContours(eye_image, contours, i, (255, 0, 0), -1)
#
# cv2.imshow("dst image", eye_image)
########################################################################################################################

# print(landmarks)            np.matrix([[eyeball_right_x,eyeball_right_y]])
# cv2.namedWindow("image", 1)
# cv2.imshow("image", cv_bgr_image)




