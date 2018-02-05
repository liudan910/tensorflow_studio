import tensorflow as tf
import os
v1 = tf.Variable(tf.constant(1.0, shape=[1], name='v1'))
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')

result = v1 + v2
init_op = tf.initialize_all_variables()

saver = tf.train.Saver()
path = "checkpoints/model1/model"
with tf.Session() as sess:

    sess.run(init_op)
    saver.save(sess, save_path=path)  #保存计算图

    """
    saver.restore(sess,path)                                 #加载已经保存的模型
    print(sess.run(result))
    """

"""
生成三个文件：
checkpoint 模型文件
model.index 每个变量的取值
model.meta 图定义
"""