# -*- coding: utf-8 -*-
import random
import urllib.request
import json
import re
import requests
import time
import urllib
import detect_Baidu
import os
import get_video
import video_face2
import division_sum
import numpy as np
import math
from xlutils.copy import copy
import xlrd


#定义要爬取的微博大V的微博ID
id=(input("请输入要抓的微博oid:"))

na='a'
#设置代理IP

iplist=['112.228.161.57:8118','125.126.164.21:34592','122.72.18.35:80','163.125.151.124:9999','114.250.25.19:80']

proxy_addr="125.126.164.21:34592"


#定义页面打开函数
def use_proxy(url,proxy_addr):
    req=urllib.request.Request(url)
    req.add_header("User-Agent","Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0")
    proxy=urllib.request.ProxyHandler({'http':random.choice(iplist)})
    opener=urllib.request.build_opener(proxy,urllib.request.HTTPHandler)
    urllib.request.install_opener(opener)
    data=urllib.request.urlopen(req).read().decode('utf-8','ignore')
    return data

#获取微博主页的containerid，爬取微博内容时需要此id
def get_containerid(url):
    data=use_proxy(url,random.choice(iplist))
    content=json.loads(data).get('data')
    for data in content.get('tabsInfo').get('tabs'):
        if(data.get('tab_type')=='weibo'):
            containerid=data.get('containerid')
    return containerid

#获取微博大V账号的用户基本信息，如：微博昵称、微博地址、微博头像、关注人数、粉丝数、性别、等级等
def get_userInfo(id):
    url='https://m.weibo.cn/api/container/getIndex?type=uid&value='+id
    data=use_proxy(url,random.choice(iplist))
    content=json.loads(data).get('data')
    profile_image_url=content.get('userInfo').get('profile_image_url')
    description=content.get('userInfo').get('description')
    profile_url=content.get('userInfo').get('profile_url')
    verified=content.get('userInfo').get('verified')
    guanzhu=content.get('userInfo').get('follow_count')
    name=content.get('userInfo').get('screen_name')
    global na
    na = name
    fensi=content.get('userInfo').get('followers_count')
    gender=content.get('userInfo').get('gender')
    urank=content.get('userInfo').get('urank')
    print("微博昵称："+name+"\n"+"微博主页地址："+profile_url+"\n"+"微博头像地址："+profile_image_url+"\n"+"是否认证："+str(verified)+"\n"+"微博说明："+description+"\n"+"关注人数："+str(guanzhu)+"\n"+"粉丝数："+str(fensi)+"\n"+"性别："+gender+"\n"+"微博等级："+str(urank)+"\n")


def download_pics(pic_url,pic_name,pic_filebagPath): #pic_url大图地址，pic_name保存图片的文件名
    pic_filePath = pic_filebagPath + '\\'
    try:
        if pic_url.endswith('.jpg'):#保存jpg图片
            f = open(pic_filePath + str(pic_name)+".jpg", 'wb')
        if pic_url.endswith('.gif'):#保存gif图片
            f = open(pic_filePath + str(pic_name)+".gif", 'wb')
        f.write((urllib.request.urlopen(pic_url)).read())
        f.close()
    except Exception as e:
        print(pic_name+" error",e)
    time.sleep(0.1)#下载间隙

# 求众数
def get_mode(L):
    x = dict((a, L.count(a)) for a in L)
    y = [k for k, v in x.items() if max(x.values()) == v]
    y = max(y)
    y = int(y)
    return y


#获取微博内容信息,并保存到文本中，内容包括：每条微博的内容、微博详情页面地址、点赞数、评论数、转发数等
def get_weibo(id):
    i=1
    k_pic = 1
    k_video = 1
    # 创建目录，不能重复创建
    os.mkdir("weibo\\{}".format(na))

    Directory = 'weibo' + '/' + na

    video_information = np.empty(shape=[0, 6])
    while True:
        url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id
        weibo_url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value=' + id + '&containerid=' + get_containerid(
            url) + '&page=' + str(i)

        # 下载图片与视频
        # try:
        data = use_proxy(weibo_url, random.choice(iplist))
        content = json.loads(data).get('data')
        cards = content.get('cards')

        for item in cards:
            item = item.get('mblog')
            try:
                title = item.get('text')
                tag = '妆'
                result = tag in title
                if result == True:
                    try:
                        date = item.get('created_at')
                        print(date)
                    except Exception as e:
                        print(e)
                        pass

                    if item:

                        # 下载视频
                        try:
                            video = item.get('page_info').get('media_info').get('stream_url')
                            if video:
                                video_url = video
                                video_name = str(na) + '_' + str(k_video)
                                k_video = k_video + 1
                                path = Directory + '/' + video_name + '.mp4'
                                try:
                                    get_video.download_file(video_url, path)  # 下载视频
                                    os.mkdir("weibo_img\\{}".format(video_name))
                                    # 计时
                                    start = time.clock()
                                    video_face2.getface(path, video_name)  # 参数为视频地址
                                    video_face2.delete_image("weibo_img\\{}".format(video_name))
                                    end = time.clock()
                                    t = end - start
                                    print("Runtime is ：", t)

                                    directory = "weibo_img\\{}".format(video_name)
                                    # 定义五官动态列表
                                    face_array = []
                                    eye_array = []
                                    eyebrow_array = []
                                    nose_array = []

                                    for root, dirs, names in os.walk(directory):
                                        for name in names:
                                            landmarks = division_sum.landmark_adjust(
                                                'weibo_img/{}'.format(video_name) + '/' + name)
                                            face_type = division_sum.face_division(landmarks)
                                            eye_type, pos1, pos2 = division_sum.eyes_division(landmarks)
                                            eyebrow_type = division_sum.eyebrows_division(landmarks, pos1, pos2)
                                            nose_type = division_sum.nose_division(landmarks)

                                            print(face_type, eye_type, eyebrow_type, nose_type)

                                            face_array.append(face_type)
                                            eye_array.append(eye_type)
                                            eyebrow_array.append(eyebrow_type)
                                            nose_array.append(nose_type)

                                    face_type = get_mode(face_array)
                                    eye_type = get_mode(eye_array)
                                    eyebrow_type = get_mode(eyebrow_array)
                                    nose_type = get_mode(nose_array)

                                    video_information = np.append(video_information,
                                                                  [[str(video_name), str(video_url), face_type, eye_type, eyebrow_type,
                                                                    nose_type]], axis=0)

                                    if k_video > 8:
                                        return video_information

                                except Exception as e:
                                    print('视频下载错误！')
                                    pass
                        except Exception as e:
                            print('该微博无视频！')
            except Exception as e:
                print(e)

            time.sleep(3)
        i += 1



if __name__=="__main__":
    get_userInfo(id)
    video_information = get_weibo(id)

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
