import codecs
import numpy as np
import logging
import time
import tensorflow as tf
import sys
import argparse
import json
from six import iteritems
from rnn_model import RNNModel

import tensorflow as tf
import logging
import time
import numpy as np


# 创建包含语料中所有词的词汇表，需要一个从字符到词汇位置索引的词典，也需要一个从位置索引到字符的词典
def create_vocab(text):
    unique_chars = list(set(text))
    print(unique_chars)
    vocab_size = len(unique_chars)
    vocab_index_dict = {}
    index_vocab_dict = {}
    for i, char in enumerate(unique_chars):
        vocab_index_dict[char] = i
        index_vocab_dict[i] = char
    return vocab_index_dict, index_vocab_dict, vocab_size


# 创建词汇后保存，后面在使用模型预测时需要读取该词汇，如果不保存而每次都创建的话则可能导致词汇顺序不同
def save_vocab(vocab_index_dict, vocab_file):
    with codecs.open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_index_dict, f, indent=2, sort_keys=True)


def load_vocab(vocab_file):
    with codecs.open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_index_dict = json.load(f)
    index_vocab_dict = {}
    vocab_size = 0
    for char, index in iteritems(vocab_index_dict):
        index_vocab_dict[index] = char
        vocab_size += 1
    return vocab_index_dict, index_vocab_dict, vocab_size


# 创建一个批量生成器用于将文本生成批量的训练样本，其中text为整个语料，batch_size为批大小，vocab_size为词汇大小，seq_length为序列长度，vocab_index_dict为词汇索引词典
class BatchGenerator(object):
    def __init__(self, text, batch_size, seq_length, vocab_size, vocab_index_dict):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.vocab_index_dict = vocab_index_dict

        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        batch = np.zeros(shape=(self._batch_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b] = self.vocab_index_dict[self._text[self._cursor[b]]]
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        batches = [self._last_batch]
        for step in range(self.seq_length):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


# 这里仅仅使用基础的RNN，直接实例化一个BasicRNNCell对象，需要指定隐含层的神经元数hidden_size。另外因为单层的RNN学习能力有限，可以设置网络的层数rnn_layers来增强神经网络的学习能力，最终用MultiRNNCell封装

# 接着给multi cell初始化，初始状态全设为0，即用multi_cell.zero_state来实现。需要传入一个batch_size参数，它会生成rnn_layers层的(batch_size ,hidden_size)个初始状态

cell_fn = tf.contrib.rnn.BasicRNNCell
cell = cell_fn(hidden_size)
cells = [cell]
for i in range(rnn_layers - 1):
    higher_layer_cell = cell_fn(hidden_size)
    cells.append(higher_layer_cell)
multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
self.zero_state = multi_cell.zero_state(self.batch_size, tf.float32)

# 开始创建占位符，有三个占位符需要创建，分别为初始状态占位符、输入占位符和target占位符。初始状态占位符主要是根据multi_cell.zero_state(self.batch_size, tf.float32)的结构使用tf.placeholder_with_default创建。输入占位符是与批大小和序列长度相关的结构[batch_size, seq_length]。target占位符结构与输入占位符一样

self.initial_state = create_tuple_placeholders_with_default(multi_cell.zero_state(self.batch_size, tf.float32),
                                                            shape=multi_cell.state_size)
self.input_data = tf.placeholder(tf.int64, [self.batch_size, self.seq_length], name='inputs')
self.targets = tf.placeholder(tf.int64, [self.batch_size, self.seq_length], name='targets')


def create_tuple_placeholders_with_default(inputs, shape):
    if isinstance(shape, int):
        result = tf.placeholder_with_default(
            inputs, list((None,)) + [shape])
    else:
        subplaceholders = [create_tuple_placeholders_with_default(
            subinputs, subshape)
            for subinputs, subshape in zip(inputs, shape)]
        t = type(shape)
        if t == tuple:
            result = t(subplaceholders)
        else:
            result = t(*subplaceholders)
    return result


# 一般需要一个嵌入层将词汇嵌入到指定的维度空间上，维度由embedding_size指定。同时vocab_size为词汇大小，这样就可以将所有单词都映射到指定的维数空间上。嵌入层通过tf.nn.embedding_lookup就能找到输入对应的词空间向量。解释下embedding_lookup操作，它会从词汇中取到inputs每个元素对应的词向量，inputs为2维的话，通过该操作后变为3维，因为已经将词用embedding_size维向量表示了

self.embedding = tf.get_variable('embedding', [vocab_size, embedding_size])
inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

# 得到的3维的嵌入层空间向量，我们无法直接传入循环神经网络，需要一些处理。根据序列长度切割，通过split后再经过squeeze操作后得到一个list，这个list就是最终要进入到循环神经网络的输入，list的长度为seq_length，它很好理解，就是有这么多个时刻的输入。每个输入的结构为(batch_size,embedding_size)，也即是(20,128)。注意这里的embedding_size，刚好也是128，与循环神经网络的隐含层神经元数量一样，这里不是巧合，而是它们必须要相同，这样嵌入层出来的矩阵输入到神经网络才能刚好与神经网络的各个权重完美相乘。最终得到循环神经网络的输出和最终状态

sliced_inputs = [tf.squeeze(input_, [1]) for input_ in
                 tf.split(axis=1, num_or_size_splits=self.seq_length, value=inputs)]
outputs, final_state = tf.contrib.rnn.static_rnn(multi_cell, sliced_inputs, initial_state=self.initial_state)

# 经过2层循环神经网络得到了输出outputs，但该输出是一个list结构，要通过tf.reshape转成tf张量形式，该张量结构为(200,128)。同样target占位符也要连接起来，结构为(200,)。接着构建softmax层，权重结构为[hidden_size, vocab_size]，偏置项结构为[vocab_size],输出矩阵与权重矩阵相乘并加上偏置项得到logits，然后使用sparse_softmax_cross_entropy_with_logits计算交叉熵损失，最后求损失平均值

flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])
flat_targets = tf.reshape(tf.concat(axis=1, values=self.targets), [-1])

softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
softmax_b = tf.get_variable("softmax_b", [vocab_size])
self.logits = tf.matmul(flat_outputs, softmax_w) + softmax_b

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=flat_targets)
mean_loss = tf.reduce_mean(loss)

# 计算平均损失，另外global_step变量用于记录训练的全局步数

count = tf.Variable(1.0, name='count')
sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')
update_loss_monitor = tf.group(sum_mean_loss.assign(sum_mean_loss + mean_loss), count.assign(count + 1),
                               name='update_loss_monitor')
with tf.control_dependencies([update_loss_monitor]):
    self.average_loss = sum_mean_loss / count
self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.0))

# 使用优化器对损失函数进行优化。为了防止梯度爆炸或梯度消失需要用clip_by_global_norm对梯度进行修正
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(mean_loss, tvars), max_grad_norm)
optimizer = tf.train.AdamOptimizer(self.learning_rate)
self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

# 创建会话开始训练，设置需要训练多少轮，由num_epochs指定。epoch_size为完整训练一遍语料库需要的轮数。运行self.zero_state得到初始状态，通过批量生成器获取一批样本数据，因为当前时刻的输入对应的正确输出为下一时刻的值，所以用data[:-1]和data[1:]得到输入和target。组织ops并将输入、target和状态对应输入到占位符上，执行

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
for i in range(num_epochs):
    model.train(session, train_size, train_batches)


def train(self, session, train_size, train_batches):
    epoch_size = train_size // (self.batch_size * self.seq_length)
    if train_size % (self.batch_size * self.seq_length) != 0:
        epoch_size += 1
    state = session.run(self.zero_state)
    start_time = time.time()
    for step in range(epoch_size):
        data = train_batches.next()
        inputs = np.array(data[:-1]).transpose()
        targets = np.array(data[1:]).transpose()
        ops = [self.average_loss, self.final_state, self.train_op, self.global_step, self.learning_rate]
        feed_dict = {self.input_data: inputs, self.targets: targets,
                     self.initial_state: state}
        average_loss, state, __, global_step, lr = session.run(ops, feed_dict)


# 前面训练的模型保存下来后就可以加载该模型进行RNN预测

module_file = tf.train.latest_checkpoint(restore_path)
model_saver.restore(session, module_file)
start_text = 'your'
length = 20
print(model.predict(session, start_text, length, vocab_index_dict, index_vocab_dict))