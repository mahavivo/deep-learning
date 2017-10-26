# from https://github.com/ankurhanda/minimal-char-rnn

import tensorflow as tf
import numpy as np

class Vocab(object):

    def __init__(self, text):
        self.text = text

        self.vocab_index_dict, self.index_vocab_dict, self.vocab_size\
            = self.create_vocab(self.text)

    def create_vocab(self, text):
        unique_chars = list(set(text))
        vocab_size = len(unique_chars)
        vocab_index_dict = {}
        index_vocab_dict = {}
        for i, char in enumerate(unique_chars):
            vocab_index_dict[char] = i
            index_vocab_dict[i] = char
        return vocab_index_dict, index_vocab_dict, vocab_size

    def char2index(self, char):
        try:
            return self.vocab_index_dict[char]
        except KeyError:
            print('Unexpected char %s', char)
        return 0

    def index2char(self, index):

        try:
            return self.index_vocab_dict[index]
        except:
            print('Unexpected index')
        return 0

    def size(self):
        return self.vocab_size


class BatchGenerator(object):

    def __init__(self, text_corpus,
                 batch_size,
                 num_unroll_steps):

        self.batch_size = batch_size
        self.text = text_corpus

        self.len_corpus = len(text_corpus)

        self.num_unroll_steps = num_unroll_steps
        self.num_batches = self.len_corpus // self.batch_size

        self.batch = np.zeros(shape=(self.batch_size, num_unroll_steps+1), dtype=np.float)
        self.last_batch = None

        self.vocab = Vocab(self.text)

        self.batch_pointer = [i*self.num_batches for i in range(0, self.batch_size)]

        # Prepare the last_batch
        _ = self.prepare_batch(num_unroll_steps=1)


    def prepare_batch(self, num_unroll_steps=None):

        if num_unroll_steps is None:
            num_unroll_steps = self.num_unroll_steps

        if self.last_batch is not None:
            self.batch[:, 0] = self.last_batch[:, 0]
        else:
            self.last_batch = np.zeros(shape=(self.batch_size, 1), dtype=np.float)

        for b in range(self.batch_size):

            seq = np.zeros(shape=(num_unroll_steps))

            for s in range(num_unroll_steps):
                seq[s] = self.vocab.char2index(self.text[self.batch_pointer[b]])
                self.batch_pointer[b] = (self.batch_pointer[b] + 1 ) % self.len_corpus

            self.last_batch[b][0] = seq[-1]

            if num_unroll_steps == self.num_unroll_steps:
                self.batch[b][1:] = seq

        return self.batch


    def get_vocab_size(self):
        return self.vocab.size()


class CustomCharRNN(object):

    def __init__(self, hidden_units, batch_size,
                 num_unroll_steps, embedding_size,
                 model, reuse, num_layers, vocab_size,
                 max_grad_norm, learning_rate):

        self.hidden_units  = hidden_units
        self.num_unrollings = num_unroll_steps
        self.batch_size = batch_size

        self.embedding_size = embedding_size
        self.model = model
        self.reuse = reuse
        self.num_layers = num_layers

        self.vocab_size = vocab_size
        self.max_grad_norm = max_grad_norm

        if self.model == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif self.model == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        elif self.model == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell


        self.input_data = tf.placeholder(tf.int64,
                                         [self.batch_size, self.num_unrollings],
                                         name='inputs')
        self.targets = tf.placeholder(tf.int64,
                                      [self.batch_size, self.num_unrollings],
                                      name='targets')
        params = {}
        if self.model == 'lstm':
            # add bias to forget gate in lstm.
            params['forget_bias'] = 0.0
            params['state_is_tuple'] = True

        cell = cell_fn(self.hidden_units,
                       reuse=tf.get_variable_scope().reuse, **params)

        cells = [cell]

        for i in range(self.num_layers - 1):
            higher_layer_cell = cell_fn(self.hidden_units, reuse=tf.get_variable_scope().reuse, **params)
            cells.append(higher_layer_cell)

        multi_cell = tf.contrib.rnn.MultiRNNCell(cells)

        with tf.name_scope('initial_state'):
            # zero_state is used to compute the intial state for cell.
            self.zero_state = multi_cell.zero_state(self.batch_size, tf.float32)

            self.initial_state = create_tuple_placeholders_with_default(
                multi_cell.zero_state(batch_size, tf.float32),
                extra_dims=(None,),
                shape=multi_cell.state_size)

        if embedding_size > 0:
            self.embedding = tf.get_variable(
                    'embedding', [self.vocab_size, self.embedding_size])
        else:
            self.embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)


        # Embeddings layers.
        with tf.name_scope('embedding_layer'):
            inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

        sliced_inputs = tf.unstack(inputs, self.num_unrollings, 1)


        # Copy cell to do unrolling and collect outputs.
        outputs, final_state = tf.contrib.rnn.static_rnn(
            multi_cell, sliced_inputs,
            initial_state=self.initial_state)

        self.final_state = final_state


        with tf.name_scope('flatten_ouputs'):
            # Flatten the outputs into one dimension.
            flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.hidden_units])

        with tf.name_scope('flatten_targets'):
            # Flatten the targets too.
            flat_targets = tf.reshape(tf.concat(axis=1, values=self.targets), [-1])

        # Create softmax parameters, weights and bias.
        with tf.variable_scope('softmax') as sm_vs:
            softmax_w = tf.get_variable("softmax_w", [self.hidden_units, self.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
            self.logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)

        with tf.name_scope('loss'):
            # Compute mean cross entropy loss for each output.
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=flat_targets)
            self.mean_loss = tf.reduce_mean(loss)

        self.global_step = tf.get_variable('global_step', [],
                                           initializer=tf.constant_initializer(0.0))

        self.learning_rate = tf.constant(learning_rate)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars),
                                          self.max_grad_norm)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                  global_step=self.global_step)



    def train_epoch(self, data_size, batch_generator, sess, divide_by_n=1):

        epoch_size = data_size // (self.batch_size * self.num_unrollings)

        state = sess.run(self.zero_state)

        sum_mean_loss = 0

        for step in range(epoch_size // divide_by_n):

            data = batch_generator.prepare_batch()
            inputs = data[:,:-1]
            targets = data[:,1:]

            ops = [self.mean_loss, self.final_state, self.train_op, self.global_step, self.learning_rate]

            feed_dict = {self.input_data: inputs, self.targets: targets,
                         self.initial_state: state}

            results = sess.run(ops, feed_dict)
            loss, state, _, global_step, lr = results

            sum_mean_loss += loss

            if step % 100 == 0 and step>0:
                sum_mean_loss = sum_mean_loss/100
                print('sum_mean_loss = ', sum_mean_loss, 'perp = ', np.exp(sum_mean_loss))
                sum_mean_loss = 0

        print('######## Finished running an epoch ###########')

    def sample_seq(self, session, length, start_text, vocab, temperature=1.0, max_prob=True):

        state = session.run(self.zero_state)

        # use start_text to warm up the RNN.
        if start_text is not None and len(start_text) > 0:
            seq = list(start_text)
            for char in start_text[:-1]:
                x = np.array([[vocab.char2index(char)]])
                state = session.run(self.final_state,
                                    {self.input_data: x,
                                     self.initial_state: state})
            x = np.array([[vocab.char2index(start_text[-1])]])
        else:
            vocab_size = len(vocab.vocab_index_dict.keys())
            x = np.array([[np.random.randint(0, vocab_size)]])
            seq = []

        for i in range(length):
            state, logits = session.run([self.final_state,
                                         self.logits],
                                        {self.input_data: x,
                                         self.initial_state: state})
            unnormalized_probs = np.exp((logits - np.max(logits)) / temperature)
            probs = unnormalized_probs / np.sum(unnormalized_probs)

            if max_prob:
                sample = np.argmax(probs[0])
            else:
                sample = np.random.choice(self.vocab_size, 1, p=probs[0])[0]

            seq.append(vocab.index2char(sample))
            x = np.array([[sample]])
        return ''.join(seq)


def create_tuple_placeholders_with_default(inputs, extra_dims, shape):
  if isinstance(shape, int):
    result = tf.placeholder_with_default(inputs, list(extra_dims) + [shape])
  else:
    subplaceholders = [create_tuple_placeholders_with_default(subinputs, extra_dims, subshape)
                       for subinputs, subshape in zip(inputs, shape)]
    t = type(shape)
    if t == tuple:
      result = t(subplaceholders)
    else:
      result = t(*subplaceholders)
  return result



import codecs
with codecs.open('./data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(text)


train_size = int(0.9 * len(text))
valid_size = int(0.1 * len(text))
test_size = len(text) - train_size - valid_size
train_text = text[:train_size]
valid_text = text[train_size:train_size + valid_size]
test_text = text[train_size + valid_size:]

bg = BatchGenerator(text_corpus=train_text,
                    batch_size=20,
                    num_unroll_steps=10)

vocab = bg.vocab

params = {'hidden_units': 128,
          'batch_size': 20,
          'num_unroll_steps': 10,
          'embedding_size': 0,
          'model': 'lstm',
          'reuse': False,
          'num_layers': 2,
          'vocab_size': bg.get_vocab_size(),
          'max_grad_norm': 5.,
          'learning_rate': 2e-3,
          }

with tf.name_scope('training'):
    charrnn_train = CustomCharRNN(**params)

tf.get_variable_scope().reuse_variables()

with tf.name_scope('sample'):
    params = {'hidden_units': 128,
              'batch_size': 1,
              'num_unroll_steps': 1,
              'embedding_size': 0,
              'model': 'lstm',
              'reuse': True,
              'num_layers': 2,
              'vocab_size': bg.get_vocab_size(),
              'max_grad_norm': 5.,
              'learning_rate': 2e-3,
              }

    charnn_sample = CustomCharRNN(**params)



start_text = "First Citizen: " \
             "Before we proceed any further, hear me speak."

num_epochs = 10


with tf.Session() as sess:

    tf.global_variables_initializer().run()

    for e in range(0, num_epochs):
        print(e)
        charrnn_train.train_epoch(train_size, bg, sess, divide_by_n=1)

    result = charnn_sample.sample_seq(sess, 1000, start_text=start_text, vocab=vocab, max_prob=False)

    print(result)