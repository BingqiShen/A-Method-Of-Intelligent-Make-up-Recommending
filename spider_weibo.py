# -*- coding: utf-8 -*-
import random
import urllib.request
import json
import re
import requests
import time
from tqdm import tqdm
from pyquery import PyQuery as pq
from urllib.parse import urlencode
import urllib
import detect_Baidu
import os
import get_video
import video_face1
import math


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




#获取微博内容信息,并保存到文本中，内容包括：每条微博的内容、微博详情页面地址、点赞数、评论数、转发数等
def get_weibo(id,file):
    i=1
    k_pic = 1
    k_video = 1
    # 创建目录，不能重复创建
    os.mkdir("weibo\\{}".format(na))

    Directory = 'weibo' + '/' + na
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
                date = item.get('created_at')
                print(date)
            except Exception as e:
                print(e)
                pass

            if item:
                # 下载图片
                # try:
                #     pics = item.get('pics')
                #     if pics:
                #         for pic in pics:
                #             picture_url = pic.get('large').get('url')  # 得到原图地址
                #             print(picture_url)
                #             pid = pic.get('pid')  # 图片id
                #             pic_name = str(na) + '_' + str(k_pic)
                #             k_pic = k_pic + 1
                #             # pic_name = timestr_standard(data['created_at']) + '_' + pid[25:]  # 构建保存图片文件名，timestr_standard是一个把微博的created_at字符串转换为‘XXXX-XX-XX’形式日期的一个函数
                #             a = detect_Baidu.face_information(picture_url)
                #
                #             try:
                #                 yaw,pitch,roll = detect_Baidu.face_angle(a)
                #                 blur,illumination = detect_Baidu.face_blur_illumination(a)
                #                 if abs(yaw) < 15 and abs(pitch) < 20 and blur < 0.7 and illumination > 80:
                #                     download_pics(picture_url, pic_name, Directory)  # 下载原图
                #                 else:
                #                     print('图片人脸不合格')
                #             except Exception as e:
                #                 print('未发现人脸！')
                #                 print(e)
                #                 pass
                # except Exception as e:
                #     print(e)
                #     print('该微博无图片！')

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
                        except Exception as e:
                            print('视频下载错误！')
                            pass
                except Exception as e:
                    print('该微博无视频！')

            time.sleep(3)
        i += 1






        # if(len(cards)>0):
        #     for j in range(len(cards)):
        #         print("-----正在爬取第"+str(i)+"页，第"+str(j)+"条微博------")
        #         card_type=cards[j].get('card_type')
        #         if(card_type==9):
        #             mblog=cards[j].get('mblog')
        #             #print(mblog)
        #             #print(str(mblog).find("转发微博"))
        #             if str(mblog).find('retweeted_status')  == -1:
        #                 if str(mblog).find('original_pic') !=-1:
        #                     img_url=re.findall(r"'url': '(.+?)'", str(mblog))##pics(.+?)
        #                     n = 1
        #                     timename = str(time.time())
        #                     timename = timename.replace('.', '')
        #                     timename = timename[7:]#利用时间作为独特的名称
        #                     for url in img_url:
        #                         print('第' + str(n) + ' 张', end='')
        #                         with open(Directory + timename+url[-5:], 'wb') as f:
        #                             f.write(requests.get(url).content)
        #                         print('...OK!')
        #                         n = n + 1
        #                     if( n%3==0 ):  ##延迟爬取，防止截流
        #                        time.sleep(3)
        #
        #
        #             attitudes_count=mblog.get('attitudes_count')
        #             comments_count=mblog.get('comments_count')
        #             created_at=mblog.get('created_at')
        #             reposts_count=mblog.get('reposts_count')
        #             scheme=cards[j].get('scheme')
        #             text=mblog.get('text')
        #             with open(file,'a',encoding='utf-8') as fh:
        #                 fh.write("----第"+str(i)+"页，第"+str(j)+"条微博----"+"\n")
        #                 fh.write("微博地址："+str(scheme)+"\n"+"发布时间："+str(created_at)+"\n"+"微博内容："+text+"\n"+"点赞数："+str(attitudes_count)+"\n"+"评论数："+str(comments_count)+"\n"+"转发数："+str(reposts_count)+"\n")
        #     i+=1
        # else:
        #     break

        # except Exception as e:
        #     print(e)
        #     pass

if __name__=="__main__":
    file='weibo\\'+id+".txt"
    get_userInfo(id)
    get_weibo(id,file)
