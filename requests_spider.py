import requests
import json
import re


url = 'https://m.bilibili.com/video/av.html'
av = 'av'
header = {'User-Agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) '
                       'AppleWebKit/537.36(KHTML, like Gecko) '
                       'Chrome/67.0.3396.99 Mobile Safari/537.36'}

params = {
        "tag":"护肤",
        "page_size":50,
        "next_offset":0,
        "platform":"pc"
    }

def get_html_text(url,headers=None):
    if None == headers:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.92 Safari/537.36"}
    response = requests.get(url,headers=headers)
    response.encoding = "utf-8"
    return response.text



def get_json(url,headers=None):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.92 Safari/537.36"}
    html = requests.get(url,params = params,headers = headers)
    # return json.loads(get_html_text(url,headers))
    return html.json()


def get_image(url,headers=None):
    if None == headers:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.92 Safari/537.36"}
    return requests.get(url,headers=headers).content

def getVedioName(url,av):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.92 Safari/537.36"}
    r = requests.get(url,headers=headers).text #取得网站代码
    #a.*?b 匹配最短的，以a开始，以b结束的字符串。
    #如果把它应用于aabab的话，它会匹配aab（第一到第三个字符）和ab（第四到第五个字符）
    text = r'<div class="index.*?src-videoPage-multiP-part-">(.*?)</div>' #正则匹配
    text1 = re.findall(text, r)#获取得到的div的内容和p标签里的内容
    desktop_path = 'C:desktop/test/'
    for i in range(0,len(text1)):#条数
        #去掉数字\d
        if(i<9):str3 = re.compile('<a href="'+av+'#page=\d"><p>#')
        else:  str3 = re.compile('<a href="'+av+'#page=\d\d"><p>#')
        title = str3.sub('',text1[i]).replace('</p></a>','')
        full_path = desktop_path + title + '.txt' #文件名称及路径
        file = open(full_path, 'w')
        file.write('<!--'+title+'-->') #往文件里写入内容
        file.close()
        # print(title + '\t\tDone')
        return title
