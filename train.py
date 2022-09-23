import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os

from os.path import join
from tqdm import tqdm
from model_CL import get_encoder, get_predictor, train_step
from data_util import load_Dataset_Original


def train_CL(
    f, 
    h, 
    dataset_one, 
    dataset_two, 
    optimizer, 
    epochs=100
    ):
    """
    训练函数
    """

    step_wise_loss = []
    epoch_wise_loss = []

    for epoch in tqdm(range(epochs)):
        for ds_one, ds_two in zip(dataset_one, dataset_two):
            loss = train_step(ds_one, ds_two, f, h, optimizer)
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))

        if epoch % 2 == 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

    return epoch_wise_loss, f, h


def Step_Original(
    model_dir, 
    load_model_name,
    output_dir, 
    save_model_name,
    epochs, 
    batch_size, 
    label_list,
    predict_model_name,
    save_predict_model_name
    ):
    """
    训练函数
    """

    # 选择显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    # 生成数据集
    dataset_one = load_Dataset_Original(
        label_list = label_list, 
        batch_size = batch_size, 
        is_oneD_Fourier = False,
        pattern = 'train'
        )
    dataset_two = load_Dataset_Original(
        label_list = label_list, 
        batch_size = batch_size, 
        is_oneD_Fourier = True,
        pattern = 'train'
        )

    # 输出模型
    get_encoder(codesize=16).summary()
    get_predictor(codesize=16).summary()

    # 设置效率参数
    decay_steps = 500
    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.01, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.6)

    # 建立模型
    f = get_encoder(codesize=16)
    h = get_predictor(codesize=16)

    # 加载模型历史权重
    if 0 < len(load_model_name) and 0 < len(predict_model_name):    
        print("load model weights...")
        f.load_weights(model_dir + load_model_name)         
        h.load_weights(model_dir + predict_model_name)

    
    # 训练
    epoch_wise_loss, f, h  = train_CL(f, h, dataset_one, dataset_two, optimizer, epochs=epochs)
    plt.plot(epoch_wise_loss)
    
    # 保存模型权重
    f.save_weights(join(output_dir, save_model_name))
    h.save_weights(join(output_dir, save_predict_model_name))

    # 绘制loss曲线
    plt.grid()
    plt.savefig('epoch_wise_loss.png')

def train_Step(Step):
    """
    三个实验的训练:
    1.只使用正常数据
    2.正常数据 + 第一个故障
    3.正常数据 + 第一，二个故障
    通过设置Step来选取进行的步骤:
    one, two, three
    """

    if 'one' == Step:
        # 只使用正常数据
        Step_Original(
            epochs = 50,
            model_dir ='', 
            load_model_name ='',
            output_dir ='code/models/Step_One', 
            save_model_name = 'Step_One_147.h5',
            label_list = [],
            predict_model_name = '',
            save_predict_model_name = 'Step_One_Predictor_147.h5',
            batch_size = 10
            )

    elif 'two' == Step:
        # 正常数据 + 第一个故障
        Step_Original(
            epochs = 50,            
            model_dir ='code/models/Step_One/', 
            load_model_name ='Step_One_147.h5',
            output_dir ='code/models/Step_Two', 
            save_model_name = 'Step_Two_147.h5',
            label_list = [1],
            predict_model_name = 'Step_One_Predictor_147.h5',
            save_predict_model_name = 'Step_Two_Predictor_147.h5',
            batch_size = 10
            )

    elif 'three' == Step:
        # 正常数据 + 第一，二个故障
        Step_Original(
            epochs = 50,
            model_dir ='code/models/Step_Two/', 
            load_model_name ='Step_Two_147.h5',
            output_dir ='code/models/Step_Three', 
            save_model_name = 'Step_Three_147.h5',
            label_list = [1, 4],
            predict_model_name = 'Step_Two_Predictor_147.h5',
            save_predict_model_name = 'Step_Three_Predictor_147.h5',
            batch_size = 10
            )

    print("suc")



if __name__ == '__main__':
    """
    训练rr
    """
    Step = 'one'   # 确定step
    train_Step(Step=Step)
