# -*- coding: utf-8 -*-
import requests
from contextlib import closing
import time


def download_file(url, path):
    with closing(requests.get(url, stream=True)) as r:
        chunk_size = 1024 * 10
        content_size = int(r.headers['content-length'])
        print('下载开始')
        with open(path, "wb") as f:
            p = ProgressData(size=content_size, unit='Kb', block=chunk_size)
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                p.output()


class ProgressData(object):

    def __init__(self, block, size, unit, file_name='', ):
        self.file_name = file_name
        self.block = block / 1000.0
        self.size = size / 1000.0
        self.unit = unit
        self.count = 0
        self.start = time.time()

    def output(self):
        self.end = time.time()
        self.count += 1
        speed = self.block / (self.end - self.start) if (self.end - self.start) > 0 else 0
        self.start = time.time()
        loaded = self.count * self.block
        progress = round(loaded / self.size, 4)
        if loaded >= self.size:
            print(u'%s下载完成\r\n' % self.file_name)
        else:
            print(u'{0}下载进度{1:.2f}{2}/{3:.2f}{4} 下载速度{5:.2%} {6:.2f}{7}/s'. \
                  format(self.file_name, loaded, self.unit, \
                         self.size, self.unit, progress, speed, self.unit))
            print('%50s' % ('/' * int((1 - progress) * 50)))


if __name__ == '__main__':
    url = ' http://pornxp.com/P-Tj132J_ch942waUVS-11hBDM9w/1902517583/360.mp4'
    path = "video/13.mp4"
    download_file(url, path)
