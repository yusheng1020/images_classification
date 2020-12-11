import numpy as np
import struct

def reading_train_data(index):# 训练集
    '''
    :param index: 读文件的序号
    :return: 每个文件的向量与标签
    '''
    file = open("data/train/f" + str(index) + ".dat", "rb")
    data_raw = struct.unpack('f' * 144 * 440, file.read(4 * 144 * 440))
    file.close()
    res_data = []
    for i in range(0, 144):
        temp = []
        for j in range(0, 440):
            temp.append(data_raw[i * 440 + j])
        res_data.append(temp)
    res_data = np.array(res_data)
    lables = []
    for i in range(0,144):
        lables.append(index)
    return res_data,lables

def reading_test_data(index):# 测试集
    '''
    :param index: 读文件的序号
    :return: 每个文件的向量与标签
    '''
    file = open("data/test/f" + str(index) + ".dat", "rb")
    data_raw = struct.unpack('f' * 18 * 440, file.read(4 * 18 * 440))
    file.close()
    res_data = []
    for i in range(0, 18):
        temp = []
        for j in range(0, 440):
            temp.append(data_raw[i * 440 + j])
        res_data.append(temp)
    res_data = np.array(res_data)
    lables = []
    for i in range(0,18):
        lables.append(index)
    return res_data,lables