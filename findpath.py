import os
import numpy as np

"""
请不要在服务器上运行本程序，本文件用于提取Librispeech的标签和路径，如需运行，请修改对应路径
"""


def read_from_path(path):
    pathname = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            pathname += [os.path.join(dirpath, filename)]
    with open('filepath2.txt', 'w') as file:
        for path in pathname:
            if path.find('.flac') > 0:
                file.write(path.replace(r'..\LibriSpeech', 'LibriSpeech') + "\n")
                # D:\yjx\xxx\xxx\LibriSpeech


def format_path(path):
    path_lists = []
    with open(path, 'r') as file:
        temp = file.readlines()
        for p in temp:
            p.replace('\\', '/')
            path_lists.append(p)

    return path_lists


def create_scp(path1, path2, path3, pl1, pl2):
    with open(path1, 'w') as file1:
        trainlist = format_path(pl1)
        file1.writelines(trainlist)

    with open(path2, 'w') as file2:
        testlist = format_path(pl2)
        file2.writelines(testlist)

    with open(path3, 'w') as file3:
        trainlist.extend(testlist)
        file3.writelines(trainlist)


def get_tab(input_path, output_path):
    """
    npy文件中是一个字典，由ndarray实现
    LibriSpeech\test\1272\128104\1272-128104-0002.flac格式提取1272作为人的标签
    Returns none
    -------
    """
    tabs = {}
    with open(input_path, 'r') as file:
        path_lists = file.readlines()
        for path in path_lists:
            temp = path.split('\\')[2]
            tabs[path.replace('\n', '')] = int(temp)

    np.save(output_path, np.array(tabs))


if __name__ == '__main__':
    read_from_path(r'xx..\LibriSpeech\train')
    read_from_path(r'xx..\LibriSpeech\test')
    create_scp('data_split/libri_tr.scp', 'data_split/libri_te.scp', 'data_split/libri_all.scp', 'filepath2.txt',
               'filepath.txt')
    get_tab('data_split/libri_all.scp', r'data_split\libri_dict.npy')
