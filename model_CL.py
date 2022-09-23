import tensorflow as tf


def get_encoder(codesize):    
    """
    获得Encoder
    """

    inputs = tf.keras.layers.Input(shape=(1024, 1))   

    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='linear')(inputs)    
    x = tf.keras.layers.BatchNormalization()(x) 
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(filters=80, kernel_size=3, strides=2, activation='linear')(x)         
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(filters=96, kernel_size=3, strides=2, activation='linear')(x)      
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(filters=112, kernel_size=3, strides=2, activation='linear')(x)        
    x = tf.keras.layers.BatchNormalization()(x)                                                     
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='linear')(x)        
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=2, activation='linear')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(filters=512, kernel_size=3, strides=2, activation='linear')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv1D(filters=1024, kernel_size=3, strides=2, activation='linear')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)         
    
    x = tf.keras.layers.Dense(units=4096, activation='relu', use_bias=False)(x)  
    x = tf.keras.layers.BatchNormalization()(x)
    z = tf.keras.layers.Dense(units=512)(x)         
    x = tf.keras.layers.BatchNormalization()(x)
    z = tf.keras.layers.Dense(units=codesize)(x)    

    f = tf.keras.Model(inputs, z)   

    return f            


def get_predictor(codesize):     
    """
    获得Predictor
    """

    inputs = tf.keras.layers.Input((codesize, ))
    x = tf.keras.layers.Dense(8, activation='relu', use_bias=False)(inputs)    
    x = tf.keras.layers.BatchNormalization()(x)
    p = tf.keras.layers.Dense(codesize)(x)    

    h = tf.keras.Model(inputs, p)    

    return h         


def loss_func(p, z):   
    """
    loss 函数
    """

    z = tf.stop_gradient(z)    
    p = tf.math.l2_normalize(p, axis=1)  
    z = tf.math.l2_normalize(z, axis=1)   
    return - tf.reduce_mean(tf.reduce_sum((p*z), axis=1))       


@tf.function
def train_step(ds_one, ds_two, f, h, optimizer):        
    """
    训练函数
    """

    with tf.GradientTape() as tape:   
        z1, z2 = f(ds_one), f(ds_two)     
        p1, p2 = h(z1), h(z2)             
        loss = loss_func(p1, z2)/2 + loss_func(p2, z1)/2   
    
    learnable_params = f.trainable_variables + h.trainable_variables   
    gradients = tape.gradient(loss, learnable_params)
    optimizer.apply_gradients(zip(gradients, learnable_params))

    return loss


if __name__ == "__name__":
    """
    测试模型
    """

    get_encoder()