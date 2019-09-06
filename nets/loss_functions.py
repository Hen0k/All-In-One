from keras import backend as K
import numpy as np
import tensorflow as tf
LAMDA = 0
SIGMOID = 3
def age_margin_mse_loss(y_true,y_pred):
    return K.max(K.square(y_pred -y_true)-2.25,0)

def age_loss(y_true,y_pred):

    global LAMDA,SIGMOID
    loss1 = (1-LAMDA) * (1.0/2.0) * K.square(y_pred - y_true)
    loss2 = LAMDA *(1 - K.exp(-(K.square(y_pred - y_true)/(2* SIGMOID))))
    return loss1+loss2
def relative_mse_loss(y_true,y_pred):
    return K.square(y_true - y_pred)/K.sqrt(y_true)

def ordlinal_loss(y_true, y_pred):
    return np.absolute(y_true - y_pred)

def encode(num):
    return np.array([1] * num + [0] * (128 - num), dtype=float)

def ordinal_activation(order, y_pred):
    order = encode(order)
    y_pred = tf.cast(y_pred, tf.float64)
    order = tf.cast(order, tf.float64)

    return tf.less(tf.reduce_sum(order),
                 tf.reduce_sum(y_pred))
    # print(y_pred.dtype)
    # print(order.dtype)

    # exit(0)
    # if tf.less(tf.reduce_sum(y_pred), tf.reduce_sum(order)):
    # tv = 0
    
    
    #     return 1
    # else:
    #     return 0
    # return tv
