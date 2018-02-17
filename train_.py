from utils import *
import numpy as np
import net
import os
import time
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # disable GPU

flags = tf.app.flags
flags.DEFINE_integer("epoch", 5, "Training epoch")
flags.DEFINE_integer("batch_size", 64, "Samples in one trainging step")
flags.DEFINE_integer("display_step", 10, "Steps to display training information")
flags.DEFINE_integer("tb_save_step", 50, "Steps to save tensorboard file")
flags.DEFINE_integer("model_save_step", 10, "Steps to save model")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")
flags.DEFINE_float("weight_decay", 0.0005, "Weight decay coefficient")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Learning rate")

flags.DEFINE_string("model", "model_1_ll",
                    "which model to train")
flags.DEFINE_string("train_data_dir", "D:\\CPIs_project\\train_data",
                    "Dir of training data")
flags.DEFINE_string("eval_data_dir", "D:\\CPIs_project\\eval_data",
                    "Dir of evaluate data")
flags.DEFINE_string("ckpt_dir", "D:\\CPIs_project\\ckpt",
                    "Dir for saving model checkpoint")
flags.DEFINE_string("tb_dir", "D:\\CPIs_project\\tb",
                    "Dir for saving tensorboard file")
FLAGS = flags.FLAGS

def main(_):
      
  train_filenames = get_filenames(FLAGS.train_data_dir)
  eval_filenames = get_filenames(FLAGS.eval_data_dir)

  filenames = tf.placeholder(tf.string, shape=[None])
  datasets = tf.data.TextLineDataset(filenames)
  # datasets = datasets.skip()
  def convert(x):
    # print(tf.size(x))
    # x = tf.substr(x, 0, tf.size(x)-1)
    # x = tf.squeeze(x)
    sparse = tf.string_split([x], ',')
    # print('1')
    dense = tf.sparse_to_dense(sparse.indices,
                               sparse.dense_shape,
                               sparse.values,
                               default_value='0')
    # print('2')
    out = tf.squeeze(dense)
    out = tf.string_to_number(out, tf.float32)
    # out = tf.cast(out, tf.float32)

    return out
  datasets = datasets.map(convert)
  datasets = datasets.shuffle(buffer_size=100, reshuffle_each_iteration=True)
  datasets = datasets.batch(FLAGS.batch_size)
  datasets = datasets.repeat()
  iterator = datasets.make_initializable_iterator()
  next_element = iterator.get_next()
  # next_c = tf.slice(next_element,
  #                   [0, 0],
  #                   [next_element.shape[0], 2640])

  # next_p = tf.slice(next_element,
  #                   [0, 2640],
  #                   [next_element.shape[0], 3000])
  # next_label = tf.slice(next_element,
  #                       [0, 5640],
  #                       [next_element.shape[0], 1])

  next_c, next_p, next_label = tf.split(next_element, [2640, 3000, 1], 1)
  next_label = tf.cast(next_label, tf.int32)
  output = getattr(net, FLAGS.model)(next_c,
                                     next_p,
                                     reuse=False,
                                     is_training=True,
                                     weight_decay=FLAGS.weight_decay,
                                     dropout_keep_prob=FLAGS.dropout_keep_prob)
  
  accuracy = tf.cast(tf.greater(tf.nn.sigmoid(output), 0.5), tf.int32)
  accuracy = tf.equal(accuracy, next_label)
  accuracy = tf.cast(accuracy, tf.int32)
  accuracy = tf.reduce_sum(accuracy)/FLAGS.batch_size

  best_eval_accuracy = 0

  loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(next_label, output))
  global_step = tf.Variable(0, name="global_step", trainable=False)
  optim = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss,
                                                               global_step=global_step)
  saver = tf.train.Saver()
  tf.summary.scalar("loss", loss)
  tf.summary.scalar("accuracy", accuracy)
  summary = tf.summary.merge_all()
  writer = tf.summary.FileWriter(FLAGS.tb_dir)

  with tf.Session() as sess:   
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    print("Initialization")
    last_ckpt = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
    if last_ckpt:
      print("loading checkpoint...")
      saver.restore(sess, last_ckpt)
    else:
      print('train from scratch...')

    for e in range(FLAGS.epoch):
      sess.run(iterator.initializer, feed_dict={filenames: train_filenames})
      # print(sess.run(next_element))
      while True:
        try:
          start_time = time.time()
          _, loss_t, accuracy_, step = sess.run([optim,
                                                loss,
                                                accuracy,
                                                global_step])
          elapsed_time = time.time() - start_time
          start_time = time.time()

          """Display training status"""
          if step % FLAGS.display_step == 0:
            print('step: %d,  total Loss: %f, accuracy: %f, secs/step: %f' % 
                  (step, loss_t, accuracy_, elapsed_time))
            # print(sess.run(output))
            # print(sess.run(next_label))
        
          """Save tensorboard file"""
          if step % FLAGS.tb_save_step == 0:
            summary_str = sess.run(summary)
            writer.add_summary(summary_str, step)
            writer.flush()
        except tf.errors.OutOfRangeError:
          break
      print('one epoch')
      sess.run(iterator.initializer, feed_dict={filenames: eval_filenames})
      print('iterator initializer success')
      nth_bs = 0
      overall_acc = 0
      while True:
        try:
          accuracy_ = sess.run(accuracy)
          overall_acc = (overall_acc*nth_bs+accuracy_)/(nth_bs+1)
          nth_bs += 1
          print(nth_bs)
        except tf.errors.OutOfRangeError:
          break
      if overall_acc > best_eval_accuracy:
        saver.save(sess, os.path.join(FLAGS.ckpt_dir, '{}_{}_{}_{}_{}_{}.ckpt'.format(
                                      FLAGS.model, FLAGS.learning_rate,
                                      FLAGS.batch_size, FLAGS.weight_decay,
                                      FLAGS.dropout_keep_prob, step//40000)))
  print(best_eval_accuracy)

if __name__ == '__main__':
  tf.app.run()
