# -*- coding:UTF-8 -*-
import sys
import requests
import time
from xlutils.copy import copy
import xlrd
import xlwt
import numpy as np
import division_sum
import video_face1
import spider_bilibili1
import os


headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.92 Safari/537.36"}
path = 'bilibili'

# 求众数
def get_mode(L):
    x = dict((a, L.count(a)) for a in L)
    y = [k for k, v in x.items() if max(x.values()) == v]
    y = max(y)
    y = int(y)
    return y


if __name__ == "__main__":
    video_information = np.empty(shape=[0, 6])
    video_id = 625984808
    try:
        # 下载视频
        global title
        title,url = spider_bilibili1.main(path,video_id)
        # 获取人脸
        if title != None and url != None:
            os.mkdir("bilibili_img\\{}".format(title))
            # 计时
            start = time.clock()
            video_face1.getface(path + '/' + title + ".flv",title)  # 参数为视频地址
            video_face1.delete_image("bilibili_img\\{}".format(title))
            end = time.clock()
            t = end - start
            print("Runtime is ：", t)
        directory = "bilibili_img\\{}".format(title)
        # 定义五官动态列表
        face_array = []
        eye_array = []
        eyebrow_array = []
        nose_array = []

        for root, dirs, names in os.walk(directory):
            for name in names:
                landmarks = division_sum.landmark_adjust('bilibili_img/{}'.format(title) + '/' + name)
                face_type = division_sum.face_division(landmarks)
                eye_type, pos1, pos2 = division_sum.eyes_division(landmarks)
                eyebrow_type = division_sum.eyebrows_division(landmarks, pos1, pos2)
                nose_type = division_sum.nose_division(landmarks)

                print(face_type,eye_type,eyebrow_type,nose_type)

                face_array.append(face_type)
                eye_array.append(eye_type)
                eyebrow_array.append(eyebrow_type)
                nose_array.append(nose_type)

        face_type = get_mode(face_array)
        eye_type = get_mode(eye_array)
        eyebrow_type = get_mode(eyebrow_array)
        nose_type = get_mode(nose_array)
        try:
            video_information = np.append(video_information,
                                      [[str(title), str(url),face_type,eye_type,eyebrow_type,nose_type]], axis=0)
        except:
            pass
    except Exception as e:
        print(e)
        pass


    data = xlrd.open_workbook('video_information.xls')
    datac = copy(data)
    table = data.sheet_by_name('Sheet1')
    rowNum = table.nrows
    shtc = datac.get_sheet(0)
    ####################################################将视频数据导入至Excel中#####################################################
    for i in range(len(video_information)):
        shtc.write(i + rowNum, 0, video_information[i, 0])
        shtc.write(i + rowNum, 1, video_information[i, 1])
        shtc.write(i + rowNum, 2, video_information[i, 2])
        shtc.write(i + rowNum, 3, video_information[i, 3])
        shtc.write(i + rowNum, 4, video_information[i, 4])
        shtc.write(i + rowNum, 5, video_information[i, 5])
    datac.save(r'video_information.xls')  # 关闭Excel文件


