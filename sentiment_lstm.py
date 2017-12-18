# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals
#%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import os 
from tensorflow.contrib.rnn import GRUCell


np.random.seed(0)

def attention(inputs, attention_size, time_major=False, return_alphas=False):
   
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

#    if time_major:
#        # (T,B,D) => (B,T,D)
#        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size,1], stddev=0.1))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
#    v = tf.tanh(tf.matmul(inputs, W_omega) + b_omega)
    v = tf.tanh(tf.reshape(tf.matmul(tf.reshape(inputs, [-1, 100]), W_omega)+ b_omega,[-1,30,attention_size]))
    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    print (v)
    
    vu = tf.reshape(tf.matmul(tf.reshape(v, [-1, attention_size]),u_omega),[-1,30])
#    vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
    print (vu)
    alphas = tf.nn.softmax(vu)              # (B,T) shape also
    print (alphas)
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    print (output)

    if not return_alphas:
        return output
    else:
        return output, alphas
    
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

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector  

tf.reset_default_graph()
class RNN(object):
    tf.set_random_seed(0)
    def _word_embedding(self, inputs, reuse=False):                
        with tf.variable_scope('word_embedding', reuse=reuse):            
            if self.W2V_paras is None:
                self.word_embedding = tf.get_variable('w', [self.vocab_size, self.input_size], initializer=self.emb_initializer)                
            else:
                self.word_embedding = tf.Variable(self.W2V_paras)     
                print('load word2vec')                       
            x = tf.nn.embedding_lookup(self.word_embedding, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x
        
    def __init__(self, hidden_size, vocab_size, input_size, output_size, nsteps, learning_rate, W2V_paras=None):
        
        self.W2V_paras = W2V_paras
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_size = self.input_size
        self.nsteps = nsteps
        self.output_size = output_size
        self.learning_rate = learning_rate    
        
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.keep_prob_ph = tf.placeholder(tf.float32)
        
        self.X = tf.placeholder(tf.int32, shape=[None, self.nsteps])
        print(self.X)
        
        self.Y = tf.placeholder(tf.int64, shape=[None])                          
        print(self.Y)
        
        self.weight_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        
        batch_size = tf.shape(self.X)[0]

        self.Why = tf.get_variable('w_o', [self.hidden_size*2, self.output_size], initializer = self.weight_initializer, dtype = tf.float32)
        self.Bhy = tf.get_variable('b_o', [self.output_size], initializer=self.const_initializer, dtype = tf.float32)                
        
        x = self._word_embedding(inputs=self.X)        
        print(x)
                        
        self.seq_len = tf.ones([batch_size], tf.int32) * self.nsteps
        
#        rnn_outputs, final_state = tf.nn.dynamic_rnn(lstm_cell_fw, x, sequence_length=seq_len,initial_state=init_state_fw)
        rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(GRUCell(self.hidden_size),
                                                                   GRUCell(self.hidden_size), x, 
                                                                   sequence_length = self.seq_len,
                                                                   dtype=tf.float32)
        # As we have Bi-LSTM, we have two output, which are not connected. So merge them
        rnn_outputs = tf.concat(axis = 2, values = rnn_outputs)
        print(rnn_outputs)
        
#        attention_output, alphas = attention(rnn_outputs, 25, return_alphas=True)    
#        drop = tf.nn.dropout(attention_output, self.keep_prob_ph)

        drop = rnn_outputs[:,-1,:]
        self.predictions = tf.matmul(drop, self.Why) + self.Bhy        
        print(self.predictions)
        
        self.pretrain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.Y))
        print(self.pretrain_loss)
        
        self.pretrain_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.pretrain_loss)        
            
    def pretrain_step(self, sess, x , y, seq_length):
        outputs = sess.run([self.pretrain_op, self.pretrain_loss, self.predictions], 
                   feed_dict={self.X: x, self.Y: y, self.keep_prob_ph:0.8, self.seq_len: seq_length})
        return outputs 
    
    def predict_step( self, sess, x, y, seq_length):        
        
        y_pred = tf.argmax(self.predictions, axis=1, name="y_proba")
        
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y, y_pred), tf.float32), name="accuracy")
        
        outputs = sess.run([self.pretrain_loss,self.accuracy], 
                    feed_dict={self.X: x, self.Y: y,self.keep_prob_ph:1, self.seq_len: seq_length})
        return outputs
        
n_epochs = 20
batch_size = 50    
learning_rate = 0.001
hidden_size = 50
input_size = 50
nsteps = 30
output_size = 2
vocab_size = len(vocab)

lstm = RNN(hidden_size, 
           vocab_size, 
           input_size, output_size, nsteps, learning_rate,paras)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_iterations_per_epoch = len(train_data) // batch_size
n_iterations_validation = len(val_data) // batch_size
best_loss_val = np.infty
checkpoint_path = "save"

mr_train = BatchGenerator(train_data, batch_size, 0)
mr_test = BatchGenerator(test_data, batch_size, 0)

with tf.Session() as sess:  
#    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
#        saver.restore(sess, checkpoint_path)
#    else:
    init.run()
            
    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):            
            X_batch, y_batch = mr_train.next()
            seq_length = nsteps * np.ones([batch_size])
            _, loss_train, _ = lstm.pretrain_step(sess,X_batch, y_batch, seq_length)
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")

        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = mr_test.next()
            seq_length = nsteps * np.ones([batch_size])
            loss_val, acc_val = lstm.predict_step(sess, X_batch, y_batch, seq_length)            
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))
              
        # And save the model if it improved:
        if loss_val < best_loss_val:
#            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val