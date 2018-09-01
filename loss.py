#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf

def spread_loss(labels, activations, margin):
    activations_shape = activations.get_shape().as_list()
    mask_t = tf.equal(labels, 1)
    mask_i = tf.equal(labels, 0)    
    activations_t = tf.reshape(
      tf.boolean_mask(activations, mask_t), [activations_shape[0], 1]
    )    
    activations_i = tf.reshape(
      tf.boolean_mask(activations, mask_i), [activations_shape[0], activations_shape[1] - 1]
    )    
    gap_mit = tf.reduce_sum(tf.square(tf.nn.relu(margin - (activations_t - activations_i))))
    return gap_mit        
def cross_entropy(y, preds):    
    y = tf.argmax(y, axis=1)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds, labels=y)                                               
    loss = tf.reduce_mean(loss) 
    return loss

def margin_loss(y, preds):    
    y = tf.cast(y,tf.float32)
    loss = y * tf.square(tf.maximum(0., 0.9 - preds)) + \
        0.25 * (1.0 - y) * tf.square(tf.maximum(0., preds - 0.1))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
#    loss = tf.reduce_mean(loss)    
    return loss