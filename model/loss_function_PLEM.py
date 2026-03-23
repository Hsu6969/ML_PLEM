import numpy as np
import tensorflow as tf

def loss_for_feedback_PLEM(y_true, y_pred):
    
    dist_weights=1.0
    gamma_weights=1.0
    final_weights=[dist_weights]+[gamma_weights]
    final_weights = tf.constant(final_weights)  # 創建常量張量
    squared_difference = tf.square(y_true - y_pred)
    weighted_squared_difference = tf.multiply(squared_difference, final_weights)  # 將權重應用到平方差上
    mean_squared_error = tf.reduce_mean(weighted_squared_difference) #計算加權平方的均值

    "下面是原學長程式碼 start"
    # mu = 0.25 # 只要改這個
    # position_weights = 1.25
    # inside_or_outside_weight = 5*mu
    # weights = [position_weights]*4+[inside_or_outside_weight] #建立一個包含四個位置權重和insideoutside權重list
    # weights = tf.constant(weights)  # 創建常量張量
    # squared_difference = tf.square(y_true - y_pred)
    # weighted_squared_difference = tf.multiply(squared_difference, weights)  # 將權重應用到平方差上
    # # p1,p2,p3,p4,i1 = weighted_squared_difference.numpy()[0]
    # mean_squared_error = tf.reduce_mean(weighted_squared_difference) #計算加權平方的均值
    # # print(f"position_loss:{(p1+p2+p3+p4)/5},inside_loss:{i1/5}")
    "下面是原學長程式碼 end"

    return mean_squared_error