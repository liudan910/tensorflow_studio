import reader
import os
import tensorflow as tf

data_path = "D:\\Project\\WorkSpace5\\data"
print(os.path.join(data_path, "ptb.train.txt"))
train_data, valid_data, test_data, _ = reader.ptb_raw_data(data_path )

print(len(train_data))
print(train_data[:100])

#实现截断并将数据组织成batch, 使用ptb_iterator
x, y = reader.ptb_producer(train_data, 4, 5)
sess = tf.InteractiveSession()

