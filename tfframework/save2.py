import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

path = "checkpoints/model2/model2"
def model():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')

    result = v1 + v2
    init_op = tf.initialize_all_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['add'])
        with tf.gfile.GFile(path,'wb') as f:
            f.write(output_graph_def.SerializeToString())

"""
    将计算图中的变量及其取值通过常量的方式保存
"""

def test():
    with tf.Session() as sess:
        with gfile.FastGFile(path,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        result = tf.import_graph_def(graph_def, return_elements=["add:0"])
        print(sess.run(result))

test()