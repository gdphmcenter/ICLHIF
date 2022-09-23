import tensorflow as tf
import numpy as np
import os
import random

from sklearn.model_selection import train_test_split
import scipy.io as scio


class MatHandler(object):    
    """
    控制数据集的类
    """

    def __init__(self, is_oneD_Fourier):    

        # Download data if needed

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.X_train_val, self.y_train_val = self.load_dataset(is_oneD_Fourier)

    def read_mat(self):
        # DONE
        data = np.array([],[])        
        label = np.array([])
        count = 0
        # 遍历oneD文件夹中的各个mat文件
        for fn in os.listdir('/code/oneD/'):            
            if fn.endswith('.mat'):                         
                # 路径
                path = '/code/oneD/'+"".join(fn)                
                read_data = scio.loadmat(path)               
                # 获得标签
                now_data_label = fn.split('_')[0]          
                # print(now_data_label)
                # 获得mat的字典列表
                var_dict = list(read_data.keys())
                # 寻找.mat文件里带DE的变量
                for var in range(len(var_dict)):        
                    check_DE = var_dict[var].split("_")
                    for check in check_DE:
                        if check == 'DE':
                            # 记录DE的位置
                            location = var
                            # 记录带DE的变量名
                            var_DE = var_dict[location]
                            break
                # 读取数据并且转置
                now_data = read_data[var_DE].T                 
                # 剔除后面
                unwanted = now_data.shape[1] %1024   
                now_data = now_data[...,:-unwanted]
                # 分割数据为1024
                now_data = now_data.reshape(-1,1024) 
                now_data_len = now_data.shape[0]        
                # 记录标签
                for layer in range(int(now_data_len)):
                    label = np.append(label, int(now_data_label))
                # 第一次记录
                if count == 0:
                    data = now_data
                    count += 1
                    continue
                # 两次以上的记录
                data = np.vstack((data,now_data))
                count += 1
        # 返回数据集的数据和标签
        data = data.reshape(-1, 1024, 1)  
        return data, label

    def load_dataset(self, is_oneD_Fourier):
        # DONE
        X, y = self.read_mat()

        # 打乱+合并 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)        
        X_train = np.squeeze(X_train)     
        X_test = np.squeeze(X_test)

        X_train = np.vstack((X_train, X_test))  
        
        y_train = y_train.reshape(-1, 1)        
        y_test = y_test.reshape(-1, 1)
        y_train = np.vstack((y_train, y_test))

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        
        X_train = X_train.reshape(-1, 1024, 1)
        X_test = X_test.reshape(-1, 1024, 1)

        # X_train包含了train + val
        X_train_val = X_train
        y_train_val = y_train

        # 将训练集train划分出来一个子集val
        # final_X_train, X_val = 914：391
        X_train, X_val = X_train[:-391], X_train[-391:]
        y_train, y_val = y_train[:-391], y_train[-391:]

        # 一维傅里叶变换
        if is_oneD_Fourier == True:
            X_train = oneD_Fourier(X_train)
            X_test = oneD_Fourier(X_test)
            X_val = oneD_Fourier(X_val)
            X_train_val = oneD_Fourier(X_train_val)    

        # 输出写成这样是为了方便，当要使用训练集，测试集，验证集的时候的修改，本实验中的
        # 原先：X_train 为训练集、X_test 为测试集、X_val包含在X_train中
        return X_train, y_train, X_val, y_val, X_test, y_test, X_train_val, y_train_val       # cheng：X_train_val为训练集 + 测试集、 X_train_val and y_train_val add by cheng


def add1(x):
    """
    不做任何处理
    """

    return x


def oneD_Fourier(data):
    """
    一维傅里叶变换
    """

    # 数据多了一维
    data = np.squeeze(data)
    # print(data.shape)
    for layer in range(data.shape[0]):
        data[layer] = abs(np.fft.fft(data[layer]))
    data = data.reshape(-1,1024,1)
    
    return data


def get_Data_By_Label(mathandler = MatHandler(is_oneD_Fourier = False), pattern = 'train_val', label_list = [1, 2, 3]):
    """
    通过标签获得数据集
    """

    # 获得总数据集
    if 'train' == pattern:
        data = mathandler.X_train
        label = mathandler.y_train
    elif 'train_val' == pattern:        #cheng
        data = mathandler.X_train_val
        label = mathandler.y_train_val
    else:
        data = mathandler.X_val
        label = mathandler.y_val

    # 分离正常数据
    idx_normal = np.where(label == 0)[0]
    data_normal = data[idx_normal]
    label_normal = label[idx_normal]

    # 按标签分离数据
    for i in label_list:
        idx = np.where(label == i)[0]
        data_temp = data[idx]
        label_temp = label[idx]
        data_normal = np.vstack((data_normal, data_temp))
        label_normal = np.hstack((label_normal, label_temp))
        # data_normal = data_temp
        # label_normal = label_temp

    # 设置随机种子, 使得数据集可以复现
    random.seed(1)

    # 打乱数据集
    index = [i for i in range(len(data_normal))]
    random.shuffle(index)
    data_normal = data_normal[index]
    label_normal = label_normal[index]

    return data_normal, label_normal


def load_Dataset_Original(
    label_list = [1,4,7], 
    batch_size = 1, 
    is_oneD_Fourier = False,
    pattern = 'train_val'
    ):


    data, labels = get_Data_By_Label(
        mathandler = MatHandler(is_oneD_Fourier = is_oneD_Fourier), 
        pattern = pattern, 
        label_list = label_list
        )
    # 获得oneD文件夹里的数据
    
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(1024).map(add1, num_parallel_calls=AUTO).batch(batch_size).prefetch(AUTO)

    return dataset


if __name__ == "__main__":
    """
    测试数据集的生成效果
    """
    # 默认pattern为'train'
    data, label = get_Data_By_Label(label_list=[])    
    
    print(data) 
    print(label)
    print(data.shape)
    print(label.shape)
    print('suc')