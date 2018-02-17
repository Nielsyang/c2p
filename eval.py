from utils import *
import numpy as np
import net
import os
import time
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # disable GPU

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "Training epoch")
flags.DEFINE_integer("batch_size", 32, "Samples in one trainging step")
flags.DEFINE_string("model", "model_1_ll",
                    "which model to train")
flags.DEFINE_string("eval_data_dir", "E:\\c2p_yy2018\\eval_data",
                    "Dir of evaluate data")
flags.DEFINE_string("ckpt_path", "E:\\c2p_yy2018\\ckpt\\...",
                    "Dir for saving model checkpoint")
FLAGS = flags.FLAGS

def main(_):
      
  eval_filenames = get_filenames(FLAGS.eval_data_dir)

  eval_c, eval_p, eval_label = batch_input( eval_filenames,
                                            FLAGS.batch_size,
                                            num_epochs=FLAGS.epoch)
  c_input_size = eval_c.shape[1]
  p_input_size = eval_p.shape[1]

  eval_output = getattr(net, FLAGS.model)(eval_c,
                                           eval_p,
                                           reuse=False,
                                           is_training=False)
  
  
  eval_accuracy = tf.cast(tf.greater(tf.nn.sigmoid(eval_output), 0.5), tf.float32)
  eval_accuracy = tf.equal(eval_accuracy, eval_label)
  eval_accuracy = tf.cast(eval_accuracy, tf.int32)
  eval_accuracy = tf.reduce_sum(eval_accuracy)/FLAGS.batch_size

  overall_acc = 0
  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    saver.restore(sess, FLAGS.ckpt_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    nth_bs = 0
    try:
      while not coord.should_stop():
        eval_acc = sess.run(eval_accuracy)
        overall_acc = (overall_acc*nth_bs+eval_acc)/(nth_bs+1)
        nth_bs += 1

    except tf.errors.OutOfRangeError: 
      print('Done evaluate -- eval accuracy: {}'.format(overall_acc))
    finally:
      coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  tf.app.run()
