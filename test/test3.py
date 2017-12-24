import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

a = tf.constant([.0,.0,.0,.0], tf.float32)
b = tf.constant([1.,2.,3.,4.], tf.float32)

result1 = tf.nn.softmax(a)
result2 = tf.nn.softmax(b)

sess = tf.Session()

print(sess.run(result1))
print(sess.run(result2))

#softmax输入是一个向量，输入后是一个归一化后的向量，用于计算向量各个值在整个向量中的权重

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
 #one_hot=True 表示用非零即1的数组保存图片表示的数值，如 图片上写的是0，内存不是直接存一个0，而是存一个数组[1,0,0,0,0,0,0], 一个图片上写的是，保存的就是[0,1,0,0,0,0,0]

#假定数据集中只有16张照片，那么y的最终结果是16*10的矩阵，每一行代表一张图片
y_ = tf.placeholder(tf.float32,[None,10])

