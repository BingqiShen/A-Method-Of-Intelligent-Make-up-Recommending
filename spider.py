import re
import requests
import os

def downloadPic(html, keyword):
    pic_url = re.findall('"middleURL":"(.*?)",', html, re.S)
    i = 0
    t = 0
    print('找到关键词:' + keyword + '的图片，现在开始下载图片...')
    for each in pic_url:
        print('正在下载第' + str(t + 1) + '张图片，图片地址:' + str(each))
        t += 1
        try:
            pic = requests.get(each, timeout=5)
        except requests.exceptions.ConnectionError:
            print('【错误】当前图片无法下载')
            continue
        string = 'pictures\\' + keyword + '_' + str(i) + '.jpg'
        # resolve the problem of encode, make sure that chinese name could be store
        fp = open(string, 'wb')
        fp.write(pic.content)
        fp.close()
        i += 1


if __name__ == '__main__':
    word = input("请输入关键词: ")
    url = 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1569663063768_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word=%E5%B0%8F%E9%BB%84%E4%BA%BA&f=3&oq=xiaohuangren&rsp=1' + word
    result = requests.get(url)
    downloadPic(result.text, word)








# import re
# import requests
#
# url = 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1530020186092_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E5%B0%8F%E9%BB%84%E4%BA%BA'
#
# html = requests.get(url).text
# pic_url = re.findall('"objURL":"(.*?)",', html, re.S)
# i = 0
# for each in pic_url:
#     print
#     each
#     try:
#         pic = requests.get(each, timeout=5)
#     except requests.exceptions.ConnectionError:
#         print
#         '【错误】当前图片无法下载'
#         continue
#     string = 'pictures/' + str(i) + '.jpg'
#     fp = open(string, 'wb')
#     fp.write(pic.content)
#     fp.close()
#     i += 1
