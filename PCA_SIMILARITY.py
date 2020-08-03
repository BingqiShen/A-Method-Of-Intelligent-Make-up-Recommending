import cv2
import numpy as np
from numpy import *
import dlib
import os
import operator

# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# 预处理函数，把图片裁剪为人脸部分，并变为30*40大小的灰度图，然后保存
# sourcefilepath是图片所在文件夹的路径，这个文件夹里有要预处理的图片
# filename是sourcefilepath文件夹里的图片的文件名
# distinationfilepath是处理后要保存到的文件夹的路径，处理后的图片保存到该文件夹并且文件名仍为filename
def PRETREAT(sourcefilepath, filename, distinationfilepath):
    img = cv2.imread(sourcefilepath + '/' + filename)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector(img, 1)

    for num, face in enumerate(faces):
        # 计算矩形大小
        # (x,y), (宽度width, 高度height)
        pos_start = tuple([face.left(), face.top()])
        pos_end = tuple([face.right(), face.bottom()])
        # 计算矩形框大小
        height = face.bottom() - face.top()
        width = face.right() - face.left()
        # 根据人脸大小生成空的图像
        img_blank = np.zeros((height, width, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                img_blank[i][j] = img[face.top() + i][face.left() + j]
        # cv2.imshow('1', img_blank)
        # cv2.waitKey(0)
        new_width = 300
        new_height = 400
        new_img = cv2.resize(img_blank, (new_width, new_height))
        # cv2.imshow('2', new_img)
        # cv2.waitKey(0)
        cv2.imwrite(distinationfilepath + '/' + filename, new_img)


# 把图片转换为向量，行向量，从第1行开始把每一行的像素值连起来
def img2vector(filename):
    img = cv2.imread(filename, 0)  # read as 'gray'
    rows, cols = img.shape
    imgVector = zeros((1, rows*cols))  # create a none vector:to raise speed
    imgVector = reshape(img, (1, rows*cols))  # change img from 2D to 1D
    return imgVector


# PCA算法的函数
# data为训练集的图片组成的矩阵，也就是备选的所有人的（多张）照片
# k为采用的特征值的数量（不一定用完所有特征值）
def pca(data, k):
    data = float32(mat(data))
    rows, cols = data.shape  # 取大小
    data_mean = mean(data, 0)  # 对列求均值
    data_mean_all = tile(data_mean, (rows, 1))  # 把对列求得的平均值行向量data_mean重复，每行一次，共rows行，与data大小一样
    Z = data - data_mean_all
    T1 = Z*Z.T  # 使用矩阵计算，所以前面mat
    D, V = linalg.eig(T1)  # 特征值与特征向量,V的第i列是对应第i个特征值的特征向量
    V1 = V[:, 0:k]  # 取前k个特征向量
    V1 = Z.T*V1
    for i in range(k):  # 特征向量归一化
        L = linalg.norm(V1[:, i])
        V1[:, i] = V1[:, i]/L

    data_new = Z*V1  # 降维后的数据
    return data_new, data_mean, V1


# 加载数据集
# 对每个人选择k（0-26）张照片作为数据集（训练集），剩下的为测试集，26是因为AR数据库有100个人，每人26张照片，大小为120*165
def loadDataSet(k):
    # step 1:Getting data set
    print("--Getting data set---")
    # note to use '/'  not '\'
    dataSetDir = 'zhihu_image'  # 数据集所在文件夹的名字
    # 显示文件夹内容
    choose = random.permutation(26)+1  # 随机排序1-26 (0-25）+1
    train_face = zeros((100*k, 120*165))
    train_face_number = zeros(100*k)
    test_face = zeros((100*(26-k), 120*165))
    test_face_number = zeros(100*(26-k))
    for i in range(100):  # 100 sample people,i=0-99
        people_num = i+1
        for j in range(23):  # everyone has 26 different face（每人26张图片）
            if j < k:
                #filename = dataSetDir+'/'+str(people_num)+'-'+str(choose[j])+'.jpg'
                filename = dataSetDir + '/' + 'shi' + str(choose[j]) + '.jpg'
                img = img2vector(filename)
                train_face[i*k+j, :] = img
                train_face_number[i*k+j] = people_num  # 记录对应的图片是哪个人的
            else:
                #filename = dataSetDir+'/'+str(people_num)+'-'+str(choose[j])+'.jpg'
                filename = dataSetDir + '/' + 'shi' + str(choose[j]) + '.jpg'
                img = img2vector(filename)
                test_face[i*(26-k)+(j-k), :] = img
                test_face_number[i*(26-k)+(j-k)] = people_num  # 记录对应的图片是哪个人的

    return train_face, train_face_number, test_face, test_face_number


# calculate the accuracy of the test_face 计算识别的准确率
def facefind():
    # Getting data set
    train_face, train_face_number, test_face, test_face_number = loadDataSet(23)  # loadDataSet的参数可以是0-26内的数
    # PCA training to train_face
    data_train_new, data_mean, V = pca(train_face, 30)
    num_train = data_train_new.shape[0]
    num_test = test_face.shape[0]
    temp_face = test_face - tile(data_mean, (num_test, 1))
    data_test_new = temp_face*V  # 得到测试脸在特征向量下的数据
    data_test_new = array(data_test_new)  # mat change to array 测试脸降维后的数据
    data_train_new = array(data_train_new)  # mat change to array
    true_num = 0
    for i in range(num_test):
        testFace = data_test_new[i, :]
        diffMat = data_train_new - tile(testFace, (num_train, 1))
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)  # 横着加，即对每行求和，对应的是该测试脸和每一个训练脸欧氏距离的平方
        sortedDistIndicies = sqDistances.argsort()
        indexMin = sortedDistIndicies[0]
        if train_face_number[indexMin] == test_face_number[i]:
            true_num += 1

    accuracy = float(true_num)/num_test
    print('The classify accuracy is: %.2f%%'%(accuracy * 100))










if __name__ == '__main__':
    # 对所有训练集图片进行预处理

    # 训练集图片所在文件夹
    sourcefilepath = "image"
    distinationfilepath = "zhihu_image"
    files = os.listdir(sourcefilepath)
    for file in files:
        if not os.path.isdir(file):
            PRETREAT(sourcefilepath, file, distinationfilepath)

    facefind()