import tensorflow as tf
import numpy as np
import random

from __feature__ import absolte_import

class Seq2SeqModel(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, max_gradient_norm, batch_size,
                 learning_rate, learning_rate_decay_factor, use_lstm  = False, num_samples = 512, forward_only = False):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)        #学习速率 变量 固定值
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        output_projection = None
        softmax_loss_function = None
        #如果样本量比词汇表的量小，那么要用抽样的softmax          ????
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w = tf.get_variable("prowj_w",[size, self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.target_vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs , labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.target_vocab_size)
                softmax_loss_function = sampled_loss
        #构建RNN
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        #Attengtion模型
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_attengtion_seq2seq(
                encoder_inputs, decoder_inputs,cell,
                num_encoder_symbols = source_vocab_size,
                num_decoder_symbols = target_vocab_size,
                embedding_size=size,
                output_projection=output_projection,
                feed_previous=do_decode
            )
        #给模型填充数据
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
        #targets的值是解码器偏移1位
        targets = [self.decoder_inputs[i+1] for i in range(len(self.decoder_inputs)-1)]
        #训练模型的输出
        if forward_only:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs,self.decoder_inputs,targets,
                self.target_weights,buckets,lambda  x,y:seq2seq_f(x, y, True),
                softmax_loss_function = softmax_loss_function)
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outpouts[b]
                    ]
        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs,targets,
                self.target_weights,buckets,
                lambda  x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function
            )
        #训练模型时，更新梯度
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms =[]
            self.updates=[]
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self.global_step
                ))
        self.saver = tf.train.Saver(tf.all_variables())

    #运行模型的每一步
    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             "%d != %d."%(len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             "%d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weighs length must be equal to the one in bucket,"
                             "%d != %d." % (len(target_weights), decoder_size))
        #输入填充
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype = np.int32)

        #输出填充,与是否向后传播有关
        if not forward_only:
            output_feed = [self.updates[bucket_id],
                           self.gradient_norms[bucket_id],
                           self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in range(decoder_size):
                output_feed.append(self.outputs[bucket_id])
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1],outputs[2],None
        else:
            return None,outputs[0], outputs[1:]

    def get_batch(self, data, bucket_id):
        """
        从指定的桶中获取一次批次的随机数据,在训练的每步(step)中使用
        data:长度为self.buckets的元组,其中每个元素都包含用于创建批次的输入和输出数据对的列表
        bucket_id:整数, 从哪个bucket获取本批次
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])























