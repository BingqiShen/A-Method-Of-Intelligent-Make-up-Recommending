import sys
import cv2
import dlib
import numpy
import numpy as np
import skimage.draw
import skimage.io
import scipy.stats


predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
load_name_0 = "candidate/yoona5.jpg"
load_name_1 = "candidate/yoona6.jpg"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

img_rd_0 = cv2.imread(load_name_0)
img_rd_1 = cv2.imread(load_name_1)
img_gray_0 = cv2.cvtColor(img_rd_0, cv2.COLOR_RGB2GRAY)
img_gray_1 = cv2.cvtColor(img_rd_1, cv2.COLOR_RGB2GRAY)
rgb_img_0 = cv2.cvtColor(img_rd_0, cv2.COLOR_BGR2RGB)
rgb_img_1 = cv2.cvtColor(img_rd_1, cv2.COLOR_BGR2RGB)
dets_0 = detector(rgb_img_0, 1)
dets_1 = detector(rgb_img_1, 1)


# 识别人脸特征点，并保存下来
faces_0 = dlib.full_object_detections()
for det in dets_0:
    faces_0.append(sp(rgb_img_0, det))

faces_1 = dlib.full_object_detections()
for det in dets_1:
    faces_1.append(sp(rgb_img_1, det))

images_0 = dlib.get_face_chips(rgb_img_0, faces_0, size=500)
# 显示计数，按照这个计数创建窗口
image_cnt_0 = 0
# 显示对齐结果
for image in images_0:
    image_cnt_0 += 1
    cv_rgb_image_0 = np.array(image).astype(np.uint8)  # 先转换为numpy数组
    cv_bgr_image_0 = cv2.cvtColor(cv_rgb_image_0, cv2.COLOR_RGB2BGR)  # opencv下颜色空间为bgr，所以从rgb转换为bgr

images_1 = dlib.get_face_chips(rgb_img_1, faces_1, size=500)
# 显示计数，按照这个计数创建窗口
image_cnt_1 = 0
# 显示对齐结果
for image in images_1:
    image_cnt_1 += 1
    cv_rgb_image_1 = np.array(image).astype(np.uint8)  # 先转换为numpy数组
    cv_bgr_image_1 = cv2.cvtColor(cv_rgb_image_1, cv2.COLOR_RGB2BGR)  # opencv下颜色空间为bgr，所以从rgb转换为bgr



def get_descriptor(load_name):
    img_rd = cv2.imread(load_name)
    rgb_img = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    #img = skimage.io.imread(load_name)
    dets = detector(rgb_img, 1)
    # 识别人脸特征点，并保存下来
    faces = dlib.full_object_detections()
    for det in dets:
        faces.append(sp(rgb_img, det))

    images = dlib.get_face_chips(rgb_img, faces, size=500)
    # 显示计数，按照这个计数创建窗口
    image_cnt = 0
    # 显示对齐结果
    for image in images:
        image_cnt += 1
        cv_rgb_image = np.array(image).astype(np.uint8)  # 先转换为numpy数组
        cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)  # opencv下颜色空间为bgr，所以从rgb转换为bgr
    img = cv2.cvtColor(cv_rgb_image, cv2.COLOR_BGR2RGB)
    assert len(dets) == 1
    shape = sp(img, dets[0])
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    face_descriptor = np.array(face_descriptor)
    assert face_descriptor.shape == (128,)
    return face_descriptor

def KL_divergence(p,q):
    return scipy.stats.entropy(p, q)

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

x0 = get_descriptor(load_name_0)
x1 = get_descriptor(load_name_1)

# 计算两个特征矩阵的欧几里得距离 d, 当 d < 0.4 时, 则认为是同一个人
d = np.linalg.norm(x0 - x1)
num = float(numpy.sum(x0*x1))
denom = np.linalg.norm(x0) * np.linalg.norm(x1)
distance_similarity = 1/(1+d)
cos_similarity=0.5+0.5*num/denom
# kl_similarity=KL_divergence(x0,x1)
# js_similarity=JS_divergence(x0,x1)
#
score=0.4*distance_similarity+0.6*cos_similarity


print('欧式距离：', d)
print('距离相似度：',distance_similarity)
print('余弦相似度：',cos_similarity)
print('整体相似度：',score)
# print('KL散度：',kl_similarity)
# print('JS散度：',js_similarity)

#cv2.imshow("img", img_rd_0)
#cv2.imshow("img", img_rd_1)

htitch=np.hstack((cv_bgr_image_0,cv_bgr_image_1))
cv2.putText(htitch, "similarity:"+str(round(score,4)*100)+'%', (550, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
cv2.imshow("img", htitch)

print(x0)
print(len(x0))
cv2.waitKey()
# cv2.destroyAllWindows()