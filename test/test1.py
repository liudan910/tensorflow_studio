import tensorflow as tf

# 图： 即操作任务
matrix1 = tf.constant([[3.,3.]])            #构建节点
matrix2 = tf.constant([[2.],
                       [2.]])
product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:          #创建会话
    #with tf.device("/gpu:1")        #可指定设备
    result  = sess.run([product])    #调用Session对象的run()方法来执行图，传入一些Tensor
    print(result)

    #会话有两个API接口： Extend和Run; Extend操作：在Graph中添加节点和边 ；Run操作：输入计算的节点和填充必要的数据后，进行运算，并输出运算结果。


state = tf.Variable(0,name='counter') #创建一个变量，初始化为标题0
input = tf.constant(3.0) #创建一个常量张量

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = input1* input2 #tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run([output],feed_dict={input1:[7.],input2:[2.]}))