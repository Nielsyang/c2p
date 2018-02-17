from utils import *
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def selu(x):
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def model(compound, protein, reuse, is_training=False,
          weight_decay=0.0005, dropout_keep_prob=0.5):
  input_ = tf.concat([compound, protein], 1)
  with tf.variable_scope('c2p', 'c2p', [input_], reuse=reuse):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      activation_fn=tf.nn.relu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out = slim.fully_connected(input_, 2048, scope='fc1')
        # out = slim.dropout(out, dropout_keep_prob, scope='dropout1')
        out = slim.fully_connected(out, 2024, scope='fc2')
        # out = slim.dropout(out, dropout_keep_prob, scope='dropout2')
        # out = slim.fully_connected(out, 512, scope='fc3')
        # out = slim.dropout(out, dropout_keep_prob, scope='dropout3')
        # out = slim.fully_connected(out, 256, scope='fc4')
        # out = slim.dropout(out, dropout_keep_prob, scope='dropout4')
        # out = slim.fully_connected(out, 128, scope='fc5')
        # out = slim.dropout(out, dropout_keep_prob, scope='dropout5')
        # out = slim.fully_connected(out, 64, scope='fc6')
        # out = slim.dropout(out, dropout_keep_prob, scope='dropout6')
        out = slim.fully_connected(out, 1, activation_fn=None, scope='logits')
        return tf.reshape(out, [tf.shape(input_)[0],])

def model_1(compound, protein, is_training=False,
            weight_decay=0.0005, dropout_keep_prob=0.5):
  with tf.variable_scope('c_net', 'c_net', [compound]):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      activation_fn=selu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out_c = slim.fully_connected(compound, 512, scope='fc1')
        out_c = slim.fully_connected(out_c, 512, scope='fc2')
        out_c = slim.fully_connected(out_c, 256, scope='fc3')
        out_c = slim.fully_connected(out_c, 128, scope='fc4')
        out_c = slim.fully_connected(out_c, 64, scope='fc5')
        # out_c = slim.fully_connected(out_c, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('p_net', 'p_net', [protein]):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      activation_fn=selu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out_p = slim.fully_connected(protein, 512, scope='fc1')
        out_p = slim.fully_connected(out_p, 512, scope='fc2')
        out_p = slim.fully_connected(out_p, 256, scope='fc3')
        out_p = slim.fully_connected(out_p, 128, scope='fc4')
        out_p = slim.fully_connected(out_p, 64, scope='fc5')
        # out_p = slim.fully_connected(out_p, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('mix_net', 'mix_net', [out_p, out_c]):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay),\
      activation_fn=selu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out = tf.concat([out_c, out_p], axis=1)
        out = slim.fully_connected(out, 64, scope='fc1')
        out = slim.fully_connected(out, 64, scope='fc2')
        out = slim.fully_connected(out, 1, activation_fn=None, scope='logits')
        return tf.reshape(out, [tf.shape(out)[0],])

def model_1_mn(compound, protein, reuse, is_training=False,
               weight_decay=0.0005, dropout_keep_prob=0.5):
  with tf.variable_scope('c_net', 'c_net', [compound], reuse=reuse):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay),\
      activation_fn=selu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out_c = slim.fully_connected(compound, 512, scope='fc1')
        out_c = slim.fully_connected(out_c, 512, scope='fc2')
        out_c = slim.fully_connected(out_c, 512, scope='fc3')
        out_c = slim.fully_connected(out_c, 512, scope='fc4')
        out_c = slim.fully_connected(out_c, 512, scope='fc5')
        # out_c = slim.fully_connected(out_c, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('p_net', 'p_net', [protein], reuse=reuse):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      activation_fn=selu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out_p = slim.fully_connected(protein, 512, scope='fc1')
        out_p = slim.fully_connected(out_p, 512, scope='fc2')
        out_p = slim.fully_connected(out_p, 512, scope='fc3')
        out_p = slim.fully_connected(out_p, 512, scope='fc4')
        out_p = slim.fully_connected(out_p, 512, scope='fc5')
        # out_p = slim.fully_connected(out_p, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('mix_net', 'mix_net', [out_p, out_c], reuse=reuse):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      activation_fn=selu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out = tf.concat([out_c, out_p], axis=1)
        out = slim.fully_connected(out, 512, scope='fc1')
        out = slim.dropout(out, dropout_keep_prob, scope='do1')
        out = slim.fully_connected(out, 512, scope='fc2')
        out = slim.dropout(out, dropout_keep_prob, scope='do2')
        out = slim.fully_connected(out, 1, activation_fn=None, scope='logits')
        return tf.reshape(out, [tf.shape(out)[0],])


def model_1_mn(compound, protein, reuse, is_training=False,
               weight_decay=0.0005, dropout_keep_prob=0.5):
  with tf.variable_scope('c_net', 'c_net', [compound], reuse=reuse):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay),\
      activation_fn=selu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out_c = slim.fully_connected(compound, 512, scope='fc1')
        out_c = slim.fully_connected(out_c, 512, scope='fc2')
        out_c = slim.fully_connected(out_c, 512, scope='fc3')
        out_c = slim.fully_connected(out_c, 512, scope='fc4')
        out_c = slim.fully_connected(out_c, 512, scope='fc5')
        # out_c = slim.fully_connected(out_c, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('p_net', 'p_net', [protein], reuse=reuse):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      activation_fn=selu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out_p = slim.fully_connected(protein, 512, scope='fc1')
        out_p = slim.fully_connected(out_p, 512, scope='fc2')
        out_p = slim.fully_connected(out_p, 512, scope='fc3')
        out_p = slim.fully_connected(out_p, 512, scope='fc4')
        out_p = slim.fully_connected(out_p, 512, scope='fc5')
        # out_p = slim.fully_connected(out_p, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('mix_net', 'mix_net', [out_p, out_c], reuse=reuse):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      activation_fn=selu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out = tf.concat([out_c, out_p], axis=1)
        out = slim.fully_connected(out, 512, scope='fc1')
        out = slim.dropout(out, dropout_keep_prob, scope='do1')
        out = slim.fully_connected(out, 512, scope='fc2')
        out = slim.dropout(out, dropout_keep_prob, scope='do2')
        out = slim.fully_connected(out, 1, activation_fn=None, scope='logits')
        return tf.reshape(out, [tf.shape(out)[0],])

def model_1_ll(compound, protein, reuse, is_training=False,
               weight_decay=0.0005, dropout_keep_prob=0.5):
  with tf.variable_scope('c_net', 'c_net', [compound], reuse=reuse):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay),\
      activation_fn=tf.nn.relu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        # compound = tf.reshape(compound, [-1, 2640])
        out_c = slim.fully_connected(compound, 1024, scope='fc1')
        out_c = slim.fully_connected(out_c, 1024, scope='fc2')
        out_c = slim.fully_connected(out_c, 1024, scope='fc3')
        # out_c = slim.fully_connected(out_c, 512, scope='fc4')
        # out_c = slim.fully_connected(out_c, 512, scope='fc5')
        # out_c = slim.fully_connected(out_c, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('p_net', 'p_net', [protein], reuse=reuse):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      activation_fn=selu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        # protein = tf.reshape(protein, [-1, 3000])
        out_p = slim.fully_connected(protein, 1024, scope='fc1')
        out_p = slim.fully_connected(out_p, 1024, scope='fc2')
        out_p = slim.fully_connected(out_p, 1024, scope='fc3')
        # out_p = slim.fully_connected(out_p, 512, scope='fc4')
        # out_p = slim.fully_connected(out_p, 512, scope='fc5')
        # out_p = slim.fully_connected(out_p, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('mix_net', 'mix_net', [out_p, out_c], reuse=reuse):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      activation_fn=selu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out = tf.concat([out_c, out_p], axis=1)
        out = slim.fully_connected(out, 512, scope='fc1')
        out = slim.dropout(out, dropout_keep_prob, scope='do1')
        out = slim.fully_connected(out, 512, scope='fc2')
        out = slim.dropout(out, dropout_keep_prob, scope='do2')
        out = slim.fully_connected(out, 1, activation_fn=None, scope='logits')
        return tf.reshape(out, [tf.shape(out)[0],])

def model_1_mn_bn(compound, protein, is_training=False,
                  weight_decay=0.0005, dropout_keep_prob=0.5):
  with tf.variable_scope('c_net', 'c_net', [compound]):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      normalizer_fn=slim.batch_norm, \
      normalizer_params={
      'decay': 0.997,
      'epsilon': 1e-5,
      'scale': True,
      }):
      with slim.arg_scope([slim.dropout, slim.batch_norm],
        is_training=is_training):
        out_c = slim.fully_connected(compound, 512, scope='fc1')
        out_c = slim.fully_connected(out_c, 512, scope='fc2')
        out_c = slim.fully_connected(out_c, 512, scope='fc3')
        out_c = slim.fully_connected(out_c, 512, scope='fc4')
        out_c = slim.fully_connected(out_c, 512, scope='fc5')
        # out_c = slim.fully_connected(out_c, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('p_net', 'p_net', [protein]):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      normalizer_fn=slim.batch_norm, \
      normalizer_params={
      'decay': 0.997,
      'epsilon': 1e-5,
      'scale': True,
      }):
      with slim.arg_scope([slim.dropout, slim.batch_norm],
        is_training=is_training):
        out_p = slim.fully_connected(protein, 512, scope='fc1')
        out_p = slim.fully_connected(out_p, 512, scope='fc2')
        out_p = slim.fully_connected(out_p, 512, scope='fc3')
        out_p = slim.fully_connected(out_p, 512, scope='fc4')
        out_p = slim.fully_connected(out_p, 512, scope='fc5')
        # out_p = slim.fully_connected(out_p, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('mix_net', 'mix_net', [out_p, out_c]):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay)):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out = tf.concat([out_c, out_p], axis=1)
        out = slim.fully_connected(out, 128, scope='fc1')
        out = slim.dropout(out, dropout_keep_prob, scope='do1')
        out = slim.fully_connected(out, 128, scope='fc2')
        out = slim.dropout(out, dropout_keep_prob, scope='do2')
        out = slim.fully_connected(out, 1, activation_fn=None, scope='logits')
        return tf.reshape(out, [tf.shape(out)[0],])

def model_2(compound, protein, is_training=False,
            weight_decay=0.0005, dropout_keep_prob=0.5):
  with tf.variable_scope('c_net', 'c_net', [compound]):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      normalizer_fn=slim.batch_norm, \
      normalizer_params={
      'decay': 0.997,
      'epsilon': 1e-5,
      'scale': True,
      }):
      with slim.arg_scope([slim.dropout, slim.batch_norm],
        is_training=is_training):
        out_c = slim.fully_connected(compound, 512, scope='fc1')
        out_c = slim.fully_connected(out_c, 256, scope='fc2')
        out_c = slim.fully_connected(out_c, 256, scope='fc3')
        out_c = slim.fully_connected(out_c, 128, scope='fc4')
        out_c = slim.fully_connected(out_c, 128, scope='fc5')
        out_c = slim.fully_connected(out_c, 64, scope='fc6')
        out_c = slim.fully_connected(out_c, 64, scope='fc7')
        # out_c = slim.fully_connected(out_c, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('p_net', 'p_net', [protein]):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      normalizer_fn=slim.batch_norm, \
      normalizer_params={
      'decay': 0.997,
      'epsilon': 1e-5,
      'scale': True,
      }):
      with slim.arg_scope([slim.dropout, slim.batch_norm],
        is_training=is_training):
        out_p = slim.fully_connected(protein, 512, scope='fc1')
        out_p = slim.fully_connected(out_c, 256, scope='fc2')
        out_p = slim.fully_connected(out_c, 256, scope='fc3')
        out_p = slim.fully_connected(out_c, 128, scope='fc4')
        out_p = slim.fully_connected(out_c, 128, scope='fc5')
        out_p = slim.fully_connected(out_c, 64, scope='fc6')
        out_p = slim.fully_connected(out_c, 64, scope='fc7')
        # out_p = slim.fully_connected(out_p, 1, activation_fn=None, scope='logits')

  with tf.variable_scope('mix_net', 'mix_net', [out_p, out_c]):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      normalizer_fn=slim.batch_norm, \
      normalizer_params={
      'decay': 0.997,
      'epsilon': 1e-5,
      'scale': True,
      }):
      with slim.arg_scope([slim.dropout, slim.batch_norm],
        is_training=is_training):
        out = tf.concat([out_c, out_p], axis=1)
        out = slim.fully_connected(out, 64, scope='fc1')
        out = slim.fully_connected(out, 64, scope='fc2')
        out = slim.fully_connected(out, 1, activation_fn=None, scope='logits')
        return tf.reshape(out, [tf.shape(out)[0],])

'''autoencoder net'''

def autoencodernet(input_, reuse, is_training=True,
                   weight_decay=0.0005, dropout_keep_prob=0.5):
  # input_ = tf.concat([compound, protein], 1)
  with tf.variable_scope('autoencoder', 'autoencoder', [input_], reuse=reuse):
    with slim.arg_scope([slim.fully_connected], 
      weights_regularizer=slim.l2_regularizer(weight_decay), \
      activation_fn=tf.nn.relu):
      with slim.arg_scope([slim.dropout],
        is_training=is_training):
        out = slim.fully_connected(input_, 1024, scope='fc1')
        out = slim.fully_connected(out, 1024, scope='fc2')
        out = slim.fully_connected(out, 1024, scope='fc3')
        out = slim.fully_connected(out, 5640, activation_fn=None, scope='logits')
        return out
