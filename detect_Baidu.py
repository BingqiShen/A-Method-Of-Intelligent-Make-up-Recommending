from aip import AipFace
import base64
import urllib
import urllib.request


def face_information(img_path):
     """ 你的 APPID AK SK """
     APP_ID = '17151694'
     API_KEY = 'PVc6kFsF0Pk8ZPpGewaIhoK8'
     SECRET_KEY = '43XAsfZnkcezXDAjA4czgDirnpa4TudW'

     client = AipFace(APP_ID, API_KEY, SECRET_KEY)

     # 打开本地文件
     f = open(r'%s' % img_path, 'rb')

     # # 打开url地址
     # f = urllib.request.urlopen(img_path)


     pic = base64.b64encode(f.read())
     image = str(pic, 'utf-8')

     imageType = "BASE64"

     """ 如果有可选参数 """
     options = {}
     options["face_field"] = "age,beauty,gender,face_shape,quality"
     options["max_face_num"] = 1
     options["face_type"] = "LIVE"

     """ 带参数调用人脸检测 """
     a = client.detect(image, imageType, options)


     # for item in a['result']['face_list']:
     #      print("年龄:", item['age'], "容貌评分:", item['beauty'], "性别:", item['gender'],"脸型：",item['face_shape'])
     return a

def face_angle(a):
     error_msg = a['error_msg']
     if error_msg == 'SUCCESS':
          result = a['result']
          face_list = result['face_list']
          for item in face_list:
             angle = item['angle']
             yaw = angle['yaw']
             pitch = angle['pitch']
             roll = angle['roll']
             return yaw, pitch, roll
     else:
          return error_msg

def face_blur_illumination(a):
     error_msg = a['error_msg']
     if error_msg == 'SUCCESS':
          result = a['result']
          face_list = result['face_list']
          for item in face_list:
             quality = item['quality']
             blur = quality['blur']
             illumination = quality['illumination']
             return blur, illumination






if __name__ == '__main__':
     # img_path = 'https://wx1.sinaimg.cn/large/4dbc6c9agy1gaqbwowsxzj20xe1e4tmz.jpg'
     img_path = '8.jpg'
     a = face_information(img_path)
     print(a)
     yaw, pitch, roll = face_angle(a)
     blur,illumination = face_blur_illumination(a)
     print(yaw,pitch,roll)
     print(blur,illumination)


