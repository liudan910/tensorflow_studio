import os
import zipfile
import tensorflow as tf
import matplotlib as plt
import collections
import random
import numpy as np
import math
filename = os.path.join("D:\workspace3\data", 'text9.zip')

# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))

vocabulary = vocabulary[0:100]
print(vocabulary)
vocabulary_size = 50000

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]  #'unk'计数初始化为-1
  print('count:', count)

  # 列表 ->元素统计  出现次数最多的n_words - 1个元素
  count.extend(collections.Counter(words).most_common(n_words - 1))
  print('count:', count)

  dictionary = dict()  #字典
  for word, _ in count:
    dictionary[word] = len(dictionary)  #为单词建立索引
  print('dictionary', dictionary)
  data = list()
  unk_count = 0  #unk
  for word in words:
    index = dictionary.get(word, 0)  # 取key=word的 value, 取不到则返回0
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  print('data', data)               # data为索引 引表
  count[0][1] = unk_count   #‘unk'计数
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  print(reversed_dictionary)
  return data, count, dictionary, reversed_dictionary

"""
0.输入单词列表words ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first']
1 单词统计count: 出现次数最多的vocabliry_size个，此时会得到按次数大小排序的 单词：统计数 列表。[['UNK', -1], ('the', 6), ('of', 3), ('used', 3), ('a', 2),]]
2 根据1得到的建立字典dictionary： 单词： 索引 {'UNK': 0, 'the': 1, 'of': 2, 'used': 3, 'a': 4}
3.根据2的字典，将words用其索引表示 data 。[8, 9, 10, 4, 5, 2, 11, 12, 3, 13, 14, 15, 16, 17, 1]
4.将字典反转 为索引：单词 {0: 'UNK', 1: 'the', 2: 'of', 3: 'used', 4: 'a', 5: 'term', 6: 'revolution', 7: 'to', 8: 'anarchism', 9: 'or

将句子 转为 单词列表，建立了 单词统计；对 统计数量排序 生成字典（单词：索引） ； 将句子用索引列表表示 ；
     将字典反转 （索引：单词）
"""
# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)

del vocabulary  # Hint to reduce memory.   #释放内存
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
#（8, 2 , 1)
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  print((batch_size // num_skips))
  for i in range(batch_size // num_skips):  #(batch_size // num_skips) = 4
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer[:] = data[:span]
      print('buffer', buffer)
      data_index = span
    else:
      buffer.append(data[data_index])  #当buffer尺寸达到maxlen时当后面加入一个元素时，会自动删除最前面的一个元素。
      data_index += 1
      print('data_index', data_index)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  print('batch ', batch)
  print('labels', labels)
  return batch, labels
"""
buffer = collection.deque(maxlen=span) span =2*skip_windows+1 =3 
此处buffer为一个双端队列，因限制了长度；
当buffer尺寸达到maxlen时当后面加入一个元素时，会自动删除最前面的一个元素。 
"""
"""
批次=8， 因每个数据重复2次。 故迭代4次，窗口为3。
batch: [14    14  5   5   2   2   8   8]
label: [[ 4][ 5][14][ 2][ 8][ 5][ 3][ 2]]
 
"""
print('开始产生批次')
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# collection集合类有 extend方法


# Step 4: Build and train a skip-gram model.  建立和训练skip-gram模型

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


graph = tf.Graph()  #图定义

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])  #输入
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) #标签
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32) #验证集


  # Ops and variables pinned to the CPU because of missing GPU implementation
  #因为缺少GPU实现，故使用cpu
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    with tf.name_scope('embeddings'):
      #嵌入矩阵
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))  #shape = [50000,128] -1到1之间的均值分布
      embed = tf.nn.embedding_lookup(embeddings, train_inputs) #len(train_inputs）*128
      """ embedding_lookup说明：
      train_inputs: [ 5   5   2   2   8   8] 存储索引 
      embedding: 假设为：10*10矩阵，值为以下：
      行 [
      0   [1,0,0,0,0,0,0,0]
      1   [0,1,0,0,0,0,0,0]
      2   [0,0,1,0,0,0,0,0]
      3   [0,0,0,1,0,0,0,0]
      4   [0,0,0,0,1,0,0,0]
        ]
      ……
      那么 embed:
      [
        [0,0,0,0,0,1,0,0,0]            # embedding[5]
        [0,0,0,0,0,1,0,0,0]            # embedding[5]
        [0,0,1,0,0,0,0,0,0]            # embedding[2]
        [0,0,1,0,0,0,0,0,0]            # embedding[2]
        [0,0,0,0,0,0,0,0,1]            # embedding[8]
        [0,0,0,0,0,0,0,0,1]            # embedding[8]
      ]
      每一个索引 index=8 对应于embedding矩阵位置index=8的向量。 
      """
      print(train_inputs)
    # Construct the variables for the NCE loss
    with tf.name_scope('weights'):
      nce_weights = tf.Variable(tf.truncated_normal(         #shape = [50000,128]
              [vocabulary_size, embedding_size],
              stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))     #shape 50000

    """
    一共有三个待训练的参数：
    embedding (?), nce_weights, nce_biases
    """

    loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    #计算minibatch样例和所有词向量的相似性

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm      #归一化 嵌入矩阵
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset) #从嵌入矩阵中查找验证数据的词向量
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True) #采用词向量计算方法，获取验证数据与所有其他词的相似性


  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph = graph) as session:
  init.run()
  print('Initialized')

  average_loss = 0
  for step in range(num_steps):
    batch_inputs, batch_labels = generate_batch(
      batch_size, num_skips, skip_window
    )
    feed_dict = {train_inputs: batch_inputs,
                 train_labels: batch_labels}

    _,loss_val = session.run([optimizer, loss],feed_dict=feed_dict)
    average_loss += loss_val

    #每隔2000步，计算一下这2000步平均每步的损失
    if step % 2000 == 0:
      if step > 0:
        average_loss /=2000
      #平均损失 ：最后2000批次的损失估计
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    if step % 10000 == 0:
      sim = similarity.eval() #eval()是变量的取值方法
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8              #最近8个数字
        nearest =  (-sim[i, :]).argsort()[1: top_k+1]  #将valid_data[i]与其他所有词的相似性排序，取最相似的的前8个。
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embedding = normalized_embeddings.eval()


#可视化向量
#描述 词向量之间路径的可视化
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] > len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(
      label, xy=(x, y),
      xytext=(5,2),
      textcoords = 'offset point',
      ha='right',
      va = 'bottom'
    )
  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join('D:\workspace3\data', 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
