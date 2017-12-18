# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals
#%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import os 
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell

tf.reset_default_graph()
np.random.seed(0)

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

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
    return tf.sqrt(squared_norm + epsilon)
    
def squash(s, axis=-1, epsilon=1e-7, name=None):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
    safe_norm = tf.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = s / safe_norm
    return squash_factor * unit_vector  

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

class RNN(object):
    tf.set_random_seed(0)

    def _routing(self, raw_weights, caps2_predicted, caps1_n_caps):
        raw_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights_round_2")        
        weighted_predictions = tf.multiply(raw_weights, caps2_predicted, name="weighted_predictions_round_2")        
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum_round_2")        
        caps2_output = squash(weighted_sum, axis=-2, name="caps2_output_round_2")
        caps2_output_tiled = tf.tile(caps2_output, [1, caps1_n_caps, 1, 1, 1], name="caps2_output_round_1_tiled")        
        
        agreement = tf.matmul(caps2_predicted, caps2_output_tiled, transpose_a=True, name="agreement")                
        
        routing_weights = tf.add(raw_weights, agreement, name="raw_weights_round_2")        
        
        return routing_weights, caps2_output
        
    def __init__(self, hidden_size, vocab_size, input_size, 
                 output_size, nsteps, learning_rate, W2V_paras=None):
        
        self.W2V_paras = W2V_paras
        self.caps1_n_caps = nsteps
        self.caps1_n_dims = hidden_size
        self.caps2_n_caps = output_size
        self.caps2_n_dims = 25
        
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nsteps = nsteps
        self.learning_rate = learning_rate
        
        self.seq_len = tf.placeholder(tf.float32, [None])
        
        self.enc_inputs = tf.placeholder(shape=(None, self.nsteps), dtype=tf.int32, name='encoder_inputs')
        self.dec_inputs = tf.placeholder(shape=(None, self.nsteps), dtype=tf.int32, name='decoder_inputs')
        self.dec_targets = tf.placeholder(shape=(None, self.nsteps), dtype=tf.int32, name='decoder_targets')
        self.mask = tf.placeholder(shape=(None, self.nsteps), dtype=tf.float32, name='mask')   

        self.Y = tf.placeholder(tf.int64, shape=[None])                          
        print(self.Y)
        
        self.weight_initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        
        batch_size = tf.shape(self.enc_inputs)[0]
        
        init_sigma = 0.01
        self.W_init = tf.random_normal(
                    shape=(1, self.caps1_n_caps, self.caps2_n_caps, self.caps2_n_dims, self.caps1_n_dims),
                    stddev=init_sigma, dtype=tf.float32, name="W_init")
        
        self.W = tf.Variable(self.W_init, name="W")
        self.W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1], name="W_tiled")
        
        word_embedding = tf.Variable(self.W2V_paras)  
        enc_embed = tf.nn.embedding_lookup(word_embedding, self.enc_inputs)
        dec_embed = tf.nn.embedding_lookup(word_embedding, self.dec_inputs)        
        
        if False:        
            c = tf.zeros([batch_size, self.hidden_size], tf.float32)
            h = tf.zeros([batch_size, self.hidden_size], tf.float32)
            lstm_cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size)
            ls = []
            for t in range(self.nsteps):
                with tf.variable_scope('lstm', reuse=(t!=0)):
                    _, (c, h) = lstm_cell_fw(inputs=enc_embed[:,t,:], state=[c, h])
                    ls.append(c)            
            rnn_outputs = tf.convert_to_tensor(ls, dtype=tf.float32)
            rnn_outputs = tf.transpose(rnn_outputs,perm=[1,0,2])
        else:            
            encoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, enc_embed,
                                                         dtype=tf.float32, time_major=True)
#            lstm_cell_fw = LSTMCell(self.hidden_size)
#            init_state_fw = lstm_cell_fw.zero_state(batch_size,dtype=tf.float32)
#            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(lstm_cell_fw, 
#                                                            inputs=enc_embed, 
#                                               initial_state=init_state_fw,
#                                               sequence_length=self.seq_len, dtype=tf.float32)
#            enc_outputs, enc_last_state = tf.nn.bidirectional_dynamic_rnn(GRUCell(self.hidden_size),
#                                                             GRUCell(self.hidden_size), enc_embed, 
#                                                             sequence_length=self.seq_len,
#                                                             dtype=tf.float32)
            decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)

            decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                decoder_cell, dec_embed, 
                initial_state=encoder_final_state,
                dtype=tf.float32, time_major=True, scope="plain_decoder")
   
            print(decoder_outputs)
            decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)  
            
            self.decoder_prediction = tf.argmax(decoder_logits, 2)
            self.recontructed_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(self.dec_targets, depth=vocab_size, dtype=tf.float32),
                    logits=decoder_logits) * self.mask)
            
            self.recontructed_loss = self.recontructed_loss /  tf.to_float(batch_size)
            print (self.recontructed_loss)

        rnn_outputs = encoder_outputs
#        rnn_outputs = tf.concat(axis = 2, values = enc_outputs)
#        rnn_outputs = tf.reduce_mean(rnn_outputs,axis=0)     
   
        if False: # check if capsule gate is enabled
            primary_capsule = []
            for t in range(self.nsteps):
                if t == 0:
                    primary_capsule.append(rnn_outputs[:,0,:])
                if t >= 1:
                    capsule_gate = tf.sigmoid(tf.multiply(enc_embed[:,t,:], rnn_outputs[:,t-1,:]))                
                    primary_capsule.append(tf.multiply(capsule_gate, rnn_outputs[:,t,:]))
            self.caps1_output = tf.transpose(primary_capsule, [1, 0, 2])
        else:
            self.caps1_output = rnn_outputs                 
            
        self.caps1_output = squash(self.caps1_output, name="caps1_output")
        print(self.caps1_output)                
        
        self.caps1_output_expanded = tf.expand_dims(self.caps1_output, -1,
                                               name="caps1_output_expanded")
        self.caps1_output_tile = tf.expand_dims(self.caps1_output_expanded, 2,
                                           name="caps1_output_tile")
        self.caps1_output_tiled = tf.tile(self.caps1_output_tile, [1, 1, self.caps2_n_caps, 1, 1],
                                     name="caps1_output_tiled")
        self.caps2_predicted = tf.nn.tanh(tf.matmul(self.W_tiled, self.caps1_output_tiled,
                                    name="caps2_predicted")) 
        
        raw_weights = tf.zeros([batch_size, self.caps1_n_caps, self.caps2_n_caps, 1, 1],
                               dtype=np.float32, name="raw_weights")        
        raw_weights, _ = self._routing(raw_weights, self.caps2_predicted, self.caps1_n_caps)
        raw_weights, _ = self._routing(raw_weights, self.caps2_predicted, self.caps1_n_caps)
        raw_weights, _ = self._routing(raw_weights, self.caps2_predicted, self.caps1_n_caps)
        raw_weights, self.caps2_output = self._routing(raw_weights, self.caps2_predicted, self.caps1_n_caps)                
                
        y_proba = safe_norm(self.caps2_output, axis=-2, name="y_proba")
        
        y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
        
        self.y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")
        print(self.y_pred)  
        
        if True: # check if margin-loss is enabled
            T = tf.one_hot(self.Y, depth=self.caps2_n_caps, name="T")
            print(T)  
          
            caps2_output_norm = safe_norm(self.caps2_output, axis=-2, keep_dims=True, name="caps2_output_norm")
            print(caps2_output_norm)
            
            present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm), name="present_error_raw")        
            present_error = tf.reshape(present_error_raw, shape=(-1, 2), name="present_error")
            print(present_error)        
            absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus), name="absent_error_raw")
            absent_error = tf.reshape(absent_error_raw, shape=(-1, 2), name="absent_error")
            print(absent_error)        
            L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L")
            print(L) 
            self.pretrain_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss") 
        else:
            self.predictions = tf.reshape(y_proba, shape=(-1, 2))
            print(self.predictions)
            self.pretrain_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.Y))
        
        alpha = 0.005
        self.pretrain_loss = tf.add(self.pretrain_loss, alpha * self.recontructed_loss, name="loss")

        self.pretrain_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.pretrain_loss)
            
    def pretrain_step(self, sess, x , y, seq_len, mask):        
        outputs = sess.run([self.pretrain_op, self.pretrain_loss, self.recontructed_loss,self.decoder_prediction], 
                   feed_dict={self.enc_inputs: x, 
                              self.dec_inputs: x,
                              self.dec_targets: x,
                              self.mask: mask,
                              self.Y: y,
                              self.seq_len:seq_len})
        return outputs 
    
    def predict_step( self, sess, x, y, seq_len, mask):        

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y, self.y_pred), tf.float64), name="accuracy")
        
        outputs = sess.run([self.pretrain_loss,self.accuracy,self.recontructed_loss,self.decoder_prediction], 
                           feed_dict={self.enc_inputs: x, 
                              self.dec_inputs: x,
                              self.dec_targets: x,
                              self.mask: mask,
                              self.Y: y,
                              self.seq_len:seq_len})
        return outputs
                
batch_size = 50    
learning_rate = 0.001
hidden_size = 25
input_size = 50
nsteps = 30
output_size = 2
vocab_size = len(vocab)
n_epochs = 10
restore_checkpoint = True

lstm = RNN(hidden_size, vocab_size, input_size, output_size, nsteps, learning_rate,paras)        

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
            
            mask_batch = np.zeros([batch_size, nsteps], dtype = np.float32)
            nonzeros = np.array(list(map(lambda x: (x != 0).sum(), X_batch))) 
            for ix, row in enumerate(mask_batch):
                row[:nonzeros[ix]] = 1
            
#            seq_length = nsteps * np.ones([batch_size])
            _, loss_train, reconstructed_loss, predict_ = lstm.pretrain_step(sess,X_batch, y_batch, nonzeros, mask_batch)
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f} {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train, reconstructed_loss),
                  end="")

        loss_vals = []
        acc_vals = []
        rec_loss_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = mr_test.next()
            
            mask_batch = np.zeros([batch_size, nsteps], dtype = np.float32)
            nonzeros = np.array(list(map(lambda x: (x != 0).sum(), X_batch))) 
            for ix, row in enumerate(mask_batch):
                row[:nonzeros[ix]] = 1
                
#            seq_length = nsteps * np.ones([batch_size])
            loss_val, acc_val, rec_loss_val, predict_ = lstm.predict_step(sess, X_batch, y_batch,  nonzeros, mask_batch)            
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            rec_loss_vals.append(rec_loss_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        rec_loss_val = np.mean(rec_loss_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f} {:.6f} {}".format(
            epoch + 1, acc_val * 100, loss_val, rec_loss_val,
            " (improved)" if loss_val < best_loss_val else ""))
        
        print('  sample {}:'.format(0))
        print('    input     > {}'.format(X_batch[0, :10]))
        print('    predicted > {}'.format(predict_[0, :10]))

        # And save the model if it improved:
        if loss_val < best_loss_val:
#            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val
