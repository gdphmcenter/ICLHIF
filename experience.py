from tkinter import X
from model_CL import get_encoder
from data_util import  get_Data_By_Label, MatHandler

from sklearn.manifold import TSNE
from  matplotlib.colors import  rgb2hex

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_with_labels(
    train, 
    labels, 
    clusters_number,
    picture_dir,
    picture_name
    ):
    """
    绘制聚类图
    """
    
    # print('ploting')
    # 设置字体
    from matplotlib import rcParams
    # 设置字体类型, 字体大小
    config = {
        "font.family":'Times New Roman',  
        "font.size":20,
    }
    rcParams.update(config)
    marker = ["o", ",", ".", "v", "^", "<", ">", "1", "2", "x"]
    fig, ax = plt.subplots()
    np.random.seed(0)
    colors = tuple([(np.random.random(),np.random.random(), np.random.random()) for i in range(clusters_number)])
    colors = [rgb2hex(x) for x in colors]  # from  matplotlib.colors import  rgb2hex

    for i, color in enumerate(colors):
        need_idx = np.where(labels == i)[0]
        ax.scatter(train[need_idx,1],train[need_idx,0], c=color, label=i, marker=marker[i], s=30)

    # plt.legend = ax.legend(loc=[1, 0])
    plt.legend(loc='right',fontsize='x-small')
    plt.xlabel('t-SNE',fontdict={'fontsize' : 20,"fontfamily":'Times New Roman'})
    plt.ylabel('t-SNE',fontdict={'fontsize' : 20,"fontfamily":'Times New Roman'})
    plt.tight_layout()
    
    # plt.title("(a)",x=0.5,y=-0.1)
    plt.savefig(picture_dir + picture_name,dpi=500)


def Experience_Original(
    codesize,
    model_dir,
    model_name,
    cluster_num,
    labels,
    picture_dir,
    picture_name,
    is_oneD_Fourier
    ):
    """
    训练只用正常数据,在三类最轻故障上画图
    """

    # 加载模型 
    f = get_encoder(codesize=codesize)
    f.load_weights(model_dir + model_name)

    # 加载数据集
    now_data, now_label = get_Data_By_Label(
        mathandler = MatHandler(is_oneD_Fourier = is_oneD_Fourier), 
        pattern = 'train_val', 
        label_list = [1, 4, 7]
        )

    # 修改标签[1, 4, 7]为[1, 2, 3]
    for i in range(len(now_label)):
        if now_label[i] == 4.0:
            now_label[i] = 2.0
        if now_label[i] == 7.0:
            now_label[i] = 3.0

    pre_data = f.predict(now_data)
    
    # TSNE投影到二维
    tsne = TSNE(perplexity=10, n_components=2, init='random', n_iter=5000)
    low_dim_embs = tsne.fit_transform(pre_data)
    labels = now_label

    # 将提取的特征存入txt,四类整合后就是Features文件夹下的0147_16dim
    with open('Features/0147_16dim.csv',mode='a+') as f:
        x0 = pre_data[np.where(labels == 0)]   
        x0_16dim = np.insert(x0,0,0,axis=1)     #第三个参数是labels对应
        np.savetxt(f, x0_16dim, fmt="%.8f", delimiter=",")
        x1 = pre_data[np.where(labels == 1)]   
        x1_16dim = np.insert(x1,0,1,axis=1)     #第三个参数是labels对应
        np.savetxt(f, x1_16dim, fmt="%.8f", delimiter=",")
        x2 = pre_data[np.where(labels == 2)]   
        x2_16dim  = np.insert(x2,0,2,axis=1)     #第三个参数是labels对应
        np.savetxt(f, x2_16dim, fmt="%.8f", delimiter=",")
        x3 = pre_data[np.where(labels == 3)]   
        x3_16dim  = np.insert(x3,0,3,axis=1)     #第三个参数是labels对应
        np.savetxt(f, x3_16dim, fmt="%.8f", delimiter=",")


    # 画图
    plot_with_labels(
        train=low_dim_embs, 
        labels=labels,
        clusters_number=cluster_num,
        picture_dir=picture_dir,
        picture_name=picture_name
        )


def experience(is_oneD_Fourier):
    """
    总实验函数
    """

    # 训练只用正常数据,在三类最轻故障上画图
    Experience_Original(
        codesize=16,
        model_dir='models/Step_One/', 
        model_name='Step_One_147.h5',
        cluster_num = 2,
        labels = [],
        picture_dir='picture/Step_One/',
        picture_name='Step_One_147.png',
        is_oneD_Fourier = is_oneD_Fourier
        )

    # 训练只用正常与最轻的第一类故障,在三类最轻故障上画图
    Experience_Original(
        codesize=16,
        model_dir='models/Step_Two/', 
        model_name='Step_Two_147.h5',
        cluster_num = 3,
        labels = [1, 2],
        picture_dir='picture/Step_Two/',
        picture_name='Step_Two_147.png',
        is_oneD_Fourier = is_oneD_Fourier
        )

    # 训练只用正常与最轻的第一，二类故障,在三类最轻故障上画图
    Experience_Original(
        codesize=16,
        model_dir='models/Step_Three/', 
        model_name='Step_Three_147.h5',
        cluster_num = 4,
        labels = [1, 2, 3],
        picture_dir='picture/Step_Three/',
        picture_name='Step_Three_147.png',
        is_oneD_Fourier = is_oneD_Fourier
        )

    print("suc")


if __name__ == "__main__":
    """
    实验
    """

    experience(is_oneD_Fourier = True)