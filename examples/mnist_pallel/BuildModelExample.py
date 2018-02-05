#讲解过程   http://www.cnblogs.com/zhouyang209117/p/6517684.html
#模型的存储与加载
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os


#定义权重参数
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

#定义模型
def model(X,w_h,w_h2,w_o,p_keep_input,p_keep_hidden):
    #第一个全连接层
    X = tf.nn.dropout(X,p_keep_input)
    h = tf.nn.relu(tf.matmul(X,w_h))

    #第二个全连接层
    h = tf.nn.dropout(h,p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h,w_h2))

    h2 = tf.nn.dropout(h2,p_keep_hidden)

    return tf.matmul(h2,w_o)

#加载数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
 #one_hot=True 表示用非零即1的数组保存图片表示的数值，如 图片上写的是0，内存不是直接存一个0，而是存一个数组[1,0,0,0,0,0,0], 一个图片上写的是，保存的就是[0,1,0,0,0,0,0]
trX,trY,teX,teY = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

X = tf.placeholder("float",[None,784])
Y = tf.placeholder("float",[None,10])

#初始化权重参数
w_h = init_weights([784,625])
w_h2 = init_weights([625,625])
w_o = init_weights([625,10])

#生成网络模型，得到预测值
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X,w_h,w_h2,w_o,p_keep_input,p_keep_hidden)

#定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
predict_op = tf.argmax(py_x,1)

#存储路径
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

#定义一个计数器，为训练轮数计数
global_step = tf.Variable(0,name='global_step', trainable=False)

#在声明完所有变量后，调用tf.train.Saver来保存和提取变量，其后面定义的变量将不会被存储
saver = tf.train.Saver()
non_storeable_variable = tf.Variable(777)

#训练并存储模型
with tf.Session() as sess:
    tf.initialize_all_variables().run() #初始化

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)   #存储所有变量

    start = global_step.eval()
    print("Start from:",start)

    for i in range(start,100):
        for start, end in zip(range(0,len(trX),128),range(128,len(trX)+1,128)):
            sess.run(train_op, feed_dict={ X:trX[start:end], Y:trY[start:end],
                                         p_keep_input: 0.8, p_keep_hidden:0.5 })

        global_step.assign(i).eval()  #用索引i 设置 且 更新(eval) global_step
        saver.save(sess, ckpt_dir +"/model.ckpt", global_step = global_step)
        print(i,np.mean(np.argmax(teY,axis=1) ==
                        sess.run(predict_op,feed_dict={
                            X:teX,
                            p_keep_input:1.0,
                            p_keep_hidden:1.0
                        })))


