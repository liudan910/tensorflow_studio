"""
LSTM
模型核心 由一个LSTM单元组成，可以在某时刻处理一个词，以及计算语句可能的延续性的概率。网络的存储状态由一个零矢量初始化并
在读取每一个词语后更新。由于计算上的原因， 以batch_size为最小批量来处理数据
伪代码示例：
lstm = rnn_cell.BasicLSTMCell(lstm_size)
初始化LSTM存储状态
state = tf.zeros([batch_size, lstm.state_size])
loss = 0.0

for current_batch_of_words in words_in_dataset:
    每次处理一批词语后更新状态值
    output, state = lstm(current_batch_of_words, state)
    LSTM输出可用于产生下一词语的预测
    logits = tf.matmul(output,softmax_w) + softmax_b
    probabilities = tf.nn.softmax(logits)
    loss += loss_function(probabilities, target_words)
截断反向传播
为使学习过程易于处理，通常做法：将向传播的梯度在（按时间）展开的步骤上照一个固定长度（num_steps)截断，通过在一次
迭代中的每个时刻上提供长度为num_steps的输入和每次迭代完成之后反射传导，这会很容易实现。
简化版：

#一次给定的迭代中的输入占位符
words = tf.placeholder(tf.int32, [batch_size, num_steps])
lstm = rnn_cell.BasicLSTMCell(lstm_size)

#初始化LSTM存储状态
initial_state = state = tf.zeros([batch_size, lstm.state_size])

for i in range(len(num_steps)):
  output,state = lstm(words[:,i],state)
  other code

final_state = state
如何实现迭代整个数据集：
  一个numpy数组，保存每一批词语之后的LSTM状态
  numpy_state = initial_state.eval()
  total_loss = 0.0
  for current_batch_of_words in words_in_dataset:
    numpy_state,current_loss = sess.run([final_state,loss])
        通过上一次迭代结果初始化LSTM状态
        feed_dict = {inital_state:numpy_state,words:current_batch_of_words})
    total_loss += current_loss
多个LSTM层堆叠 ：
  lstm = rnn_cell.BasicLSTMCell(lstm_size)
  stacked_lstm = rnn_cell.MultiRNNCell([lstm]*number_of_layers)
  initial_sate = state = stacked_lstm.zero_state(batch_size, tf.float32)
  for i in range(len(num_steps)):
   output,state = stacked_lstm(words[:,i],state)
   other code
final_state  = state

"""






























