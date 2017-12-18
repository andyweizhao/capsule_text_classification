from __future__ import division, print_function, unicode_literals
import numpy as np
import tensorflow as tf

tf.reset_default_graph()
sess = tf.InteractiveSession()

class BatchGenerator(object):
    """Generate and hold batches."""
    def __init__(self, dataset, batch_size,input_size):
      self._dataset = dataset
      self._batch_size = batch_size    
      self._cursor = 0      
      self._input_size = input_size      
      
      index = np.arange(len(self._dataset))
      np.random.shuffle(index)
      self._dataset = self._dataset[index]
      
    def next(self):
      if self._cursor + self._batch_size > len(self._dataset):
          self._cursor = 0
      """Generate a single batch from the current cursor position in the data."""      
      batch = self._dataset[self._cursor : self._cursor + self._batch_size,:]
      self._cursor += self._batch_size
      return batch[:,1:], batch[:,0]  

input_embedding_size = 50
batch_size = 50
encoder_hidden_units = 50
decoder_hidden_units = 50

vocab_size = len(vocab)

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
mask = tf.placeholder(shape=(None, None), dtype=tf.float32, name='mask')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                         dtype=tf.float32, time_major=True)
del encoder_outputs

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded, 
    initial_state=encoder_final_state,
    dtype=tf.float32, time_major=True, scope="plain_decoder")

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)

decoder_prediction = tf.argmax(decoder_logits, 2)

stepwise_cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
                    logits=decoder_logits) * mask)
loss = stepwise_cross_entropy / batch_size        

train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

n_iterations_per_epoch = len(train_data) // batch_size

mr_train = BatchGenerator(train_data, batch_size, 0)

for epoch in range(n_epochs):
    for iteration in range(1, n_iterations_per_epoch + 1):            
        X_batch, y_batch = mr_train.next()        

        mask_batch = np.zeros([batch_size, nsteps], dtype = np.float32)
        nonzeros = np.array(list(map(lambda x: (x != 0).sum(), X_batch))) 
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
                
        fd = {encoder_inputs: X_batch,
              decoder_inputs: X_batch,
              decoder_targets: X_batch,
              mask: mask_batch}
        _, loss_train, predict_ = sess.run([train_op, loss, decoder_prediction], fd)
        print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
          iteration, n_iterations_per_epoch,
          iteration * 100 / n_iterations_per_epoch,
          loss_train),
      end="")
    for i, (inp, pred) in enumerate(zip(fd[encoder_inputs], predict_)):
        print('  sample {}:'.format(i + 1))
        print('    input     > {}'.format(inp[:10]))
        print('    predicted > {}'.format(pred[:10]))
        if i >= 2:
            break
        print()