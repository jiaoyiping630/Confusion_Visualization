#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   把数据存在指定的pkl文件中
def save_variables(var_list, target_path, override=True):
    from .files import create_dir
    import pickle
    import os
    #   若文件存在且不覆写，直接返回
    if os.path.isfile(target_path) and not override:
        return
    #   如果目标路径的文件夹不存在，先创建
    try:
        folder_path, _ = os.path.split(target_path)
        create_dir(folder_path)
    except:
        pass
    #   然后保存数据
    if not isinstance(var_list, list):
        var_list = [var_list]
    pickle_file = open(target_path, 'wb')
    for item in var_list:
        pickle.dump(item, pickle_file)
    pickle_file.close()


#   从指定的pkl文件中读取数据
def load_variables(target_path):
    import pickle
    return_list = []
    pickle_file = open(target_path, 'rb')
    while True:
        try:
            item = pickle.load(pickle_file)
            return_list.append(item)
        except:
            break
    pickle_file.close()
    return return_list
