# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os


#   获取一个文件夹下所有的文件，排序后形成路径的list
def get_file_list(path, include_string=None, ext=''):
    import os
    if include_string is None:
        return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)])
    else:
        return sorted(
            [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext) and f.find(include_string) > -1])


#   创建一个文件夹
def create_dir(path):
    if path == '':
        return
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except:
        pass
