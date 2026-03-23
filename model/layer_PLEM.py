import tensorflow as tf
import numpy as np
from tensorflow.keras.constraints import MinMaxNorm

class Feedbacklayer_PLEM(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
      super(Feedbacklayer_PLEM, self).__init__(**kwargs)
      self.units = units

    "建立layer的權重參數"
    def build(self, input_shape):
        # 定義權重參數
        self.kernel = self.add_weight("kernel", shape=(input_shape[0][-1], self.units),
        initializer="random_normal", trainable=True)
        # # 定義偏置參數a
        self.bias_a = self.add_weight("bias_a", shape=(self.units,),initializer="zeros", trainable=True)
        # 定義偏置參數b
        self.bias_b = self.add_weight("bias_b", shape=(self.units,),initializer="zeros",trainable=True,constraint=MinMaxNorm(min_value=0, max_value=10**-5))
        super(Feedbacklayer_PLEM, self).build(input_shape)

    "建立layer的前向傳播邏輯"
    def call(self, inputs):
      "分別取得兩個輸入張量"
      input1, input2 = inputs
      "從input2中提取三個子向量"
      input2_1=input2[0][0:2]
      input2_2=input2[0][2:4]
      input2_3=input2[0][4:6]
      "將這三個子向量相加"
      input2=tf.add(tf.add(input2_1,input2_2),input2_3)
      "將結果使用ReLU activation function 然後除3"
      input2=tf.nn.relu(input2)/3

      return tf.matmul(input1, self.kernel) + self.bias_a + input2*self.bias_b*0.0001
      # return tf.matmul(input1, self.kernel) + self.bias_b
    
    "取得layer config"
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config