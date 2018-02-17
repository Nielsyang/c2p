 # -*- coding: utf-8 -*- 
import numpy as np
import tensorflow as tf
import os
import pandas as pd

def get_filenames(dir):
  filenames = []
  for rt,dirs,files in os.walk(dir):
    for file in files:
      filenames.append(os.path.join(rt,file))
  return filenames

def get_eval_data(eval_filenames):
  np_data_lists = []
  for filename in eval_filenames:
    np_data_lists.append(pd.read_csv(filename, sep=',').values)
  eval_data = np.concatenate(tuple(np_data_lists), axis=0)
  return eval_data

def batch_input(filenames, batch_size, num_epochs=None, shuffle=True):
  filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
  reader = tf.TextLineReader()
  _, value = reader.read(filename_queue)
  data = tf.decode_csv(value,[[0.0] for i in range(5641)])
  example_c = tf.stack(data[:2640])
  example_p = tf.stack(data[2640:-1])
  label = tf.stack(data[-1])

  min_after_dequeue = 128
  capacity = min_after_dequeue + 3 * batch_size
  if shuffle:
    example_batch_c, example_batch_p, label_batch = tf.train.shuffle_batch(
        [example_c, example_p, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
  else:
    example_batch_c, example_batch_p, label_batch = tf.train.batch(
        [example_c, example_p, label], batch_size=batch_size, capacity=capacity,
         allow_smaller_final_batch=True)   
  return example_batch_c, example_batch_p, label_batch