import cv2
import dlib
import numpy as np
import time
import os
import glob
import detect_Baidu


def getface(path,title):
  cap = cv2.VideoCapture(path)
  detector = dlib.get_frontal_face_detector()

  suc = cap.isOpened()  # 是否成功打开
  frame_count = 0
  out_count = 0

  skip_step = 100 #间隔100帧再截图
  skip_count = 0  #间隔计数器
  face_flag = 0   #是否截到了人脸
  face_continue_cut = 0 #连续截到人脸次数

  while suc:
      frame_count += 1
      print("正在处理第{}帧".format(frame_count))
      if out_count >= 10: #最多取出多少张
        break

      skip_count += 1

      suc, frame = cap.read() #读取一帧
      params = []
      params.append(2)  # params.append(1)
      if skip_count == skip_step or (face_flag == 1 and face_continue_cut < 5):
          skip_count = 0
          rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # opencv的颜色空间是BGR，需要转为RGB才能用在dlib中
          faceRects = detector(rgb_img, 1)

          if len(faceRects) == 1:          #只检测到一张人脸
            face_flag = 1
            face_continue_cut += 1
            for faceRect in faceRects:  #单独框出每一张人脸

                # x, y, w, h = faceRect
                # image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转为灰度图
                # img_new = cv2.resize(image,(47,57), interpolation = cv2.INTER_CUBIC) #处理面部的大小
                cv2.imwrite('bilibili_img' + '/' + title + '/' + '{}.jpg'.format(out_count + 1), frame)  # 存储到指定目录
                cv2.imencode('.jpg', frame)[1].tofile('bilibili_img' + '/' + title + '/' + '{}.jpg'.format(out_count + 1))
                # beauty.bea(r'zhihu_image/{}.jpg'.format(out_count))
                out_count += 1
                print('成功提取'+ title +'的第%d个脸部'%out_count)
                break #每帧只获取一张脸，删除这个即为读出全部面部
          else:
              face_flag = 0
              face_continue_cut = 0
  cap.release()
  cv2.destroyAllWindows()
  print('总帧数:', frame_count)
  print('提取脸部:',out_count)

def delete_image(path):
    paths = glob.glob(os.path.join(path, '*.jpg'))
    for file in paths:
        a = detect_Baidu.face_information(file)
        try:
            yaw, pitch, roll = detect_Baidu.face_angle(a)
            blur, illumination = detect_Baidu.face_blur_illumination(a)
            if abs(yaw) > 15 or abs(pitch) > 20 or blur > 0.7 or illumination < 80:
                os.remove(file)
        except Exception as e:
            print(e)
            pass
        # fp = open(file, 'rb')
        # img = Image.open(fp)
        # fp.close()




if __name__ == '__main__':
    # name = 'aishang'
    # os.mkdir("zhihu_image\\{}".format(name))

    path = 'D:\\bli'

    for root, dirs, names in os.walk(path):

        for name in names:

            ext = os.path.splitext(name)[1]  # 获取后缀名
            if ext == '.flv':
                str_end = name.find('.flv')
                str_name = name[:str_end]

                os.mkdir("bilibili_img\\{}".format(str_name))
                # 计时

                start = time.clock()
                getface(path+'/'+name) #参数为视频地址
                delete_image("bilibili_image\\{}".format(str_name))
                end = time.clock()
                t = end - start
                print("Runtime is ：", t)