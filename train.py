from utils import *
import numpy as np
import net
import os
import time
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # disable GPU

flags = tf.app.flags
flags.DEFINE_integer("epoch", 5, "Training epoch")
flags.DEFINE_integer("batch_size", 64, "Samples in one trainging step")
flags.DEFINE_integer("display_step", 10, "Steps to display training information")
flags.DEFINE_integer("tb_save_step", 50, "Steps to save tensorboard file")
flags.DEFINE_integer("model_save_step", 200000, "Steps to save model")
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

  train_c, train_p, train_label = batch_input(train_filenames,
                                              FLAGS.batch_size,
                                              num_epochs=FLAGS.epoch)
  c_input_size = train_c.shape[1]
  p_input_size = train_p.shape[1]

  # eval_c = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, c_input_size))
  # eval_p = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, p_input_size))
  # eval_label = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, ))

  # eval_data = get_eval_data(eval_filenames)
  # eval_data_size = eval_data.shape[0]

  train_output = getattr(net, FLAGS.model)(train_c,
                                           train_p,
                                           reuse=False,
                                           is_training=True,
                                           weight_decay=FLAGS.weight_decay,
                                           dropout_keep_prob=FLAGS.dropout_keep_prob)
  
  # eval_output = getattr(net, FLAGS.model)(eval_c, eval_p, reuse=True,
  #                                         is_training=False)
  
  train_accuracy = tf.cast(tf.greater(tf.nn.sigmoid(train_output), 0.5), tf.float32)
  train_accuracy = tf.equal(train_accuracy, train_label)
  train_accuracy = tf.cast(train_accuracy, tf.int32)
  train_accuracy = tf.reduce_sum(train_accuracy)/FLAGS.batch_size
  # eval_accuracy = tf.cast(tf.greater(tf.nn.sigmoid(eval_output), 0.5), tf.float32)
  # eval_accuracy = tf.equal(eval_accuracy, eval_label)
  # eval_accuracy = tf.cast(eval_accuracy, tf.int32)
  # eval_accuracy = tf.reduce_sum(eval_accuracy)/FLAGS.batch_size

  # best_eval_accuracy = 0

  train_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(train_label, train_output))
  # eval_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(eval_label, eval_output))
  global_step = tf.Variable(0, name="global_step", trainable=False)
  optim = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(train_loss,
                                                               global_step=global_step)
  saver = tf.train.Saver()
  tf.summary.scalar("train_loss", train_loss)
  tf.summary.scalar("train_accuracy", train_accuracy)
  summary = tf.summary.merge_all()
  writer = tf.summary.FileWriter(FLAGS.tb_dir)

  # eval_status = pd.DataFrame()
  # eval_status_steps = []
  # eval_status_accuracy = []
  # eval_status_loss = []

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

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    start_time = time.time()
    try:
      while not coord.should_stop():
        _, loss_t, accuracy, step = sess.run([optim,
                                              train_loss,
                                              train_accuracy,
                                              global_step])
        elapsed_time = time.time() - start_time
        start_time = time.time()

        """Display training status"""
        if step % FLAGS.display_step == 0:
          print('step: %d,  total Loss: %f, accuracy: %f, secs/step: %f' % 
                (step, loss_t, accuracy, elapsed_time))
        
        """Save tensorboard file"""
        if step % FLAGS.tb_save_step == 0:
          summary_str = sess.run(summary)
          writer.add_summary(summary_str, step)
          writer.flush()
        
        """Save model"""
        if step % FLAGS.model_save_step == 0:
          # offset = 0
          # eval_step = 0
          # eval_acc_all = 0
          # eval_loss_all = 0

          # while offset + FLAGS.batch_size < eval_data_size:
          #   eval_acc_batch, eval_loss_batch = sess.run(
          #     [eval_accuracy, eval_loss],
          #     feed_dict={
          #       eval_c:eval_data[offset:offset+FLAGS.batch_size, :c_input_size],
          #       eval_p:eval_data[offset:offset+FLAGS.batch_size,
          #                c_input_size:c_input_size+p_input_size],
          #       eval_label:eval_data[offset:offset+FLAGS.batch_size, -1]})

          #   eval_acc_all = (eval_acc_all*eval_step+eval_acc_batch)/(eval_step+1)
          #   eval_loss_all = (eval_loss_all*eval_step+eval_loss_batch)/(eval_step+1)
          #   offset += FLAGS.batch_size
          #   eval_step += 1
          
          # eval_status_steps.append(step)
          # eval_status_accuracy.append(eval_acc_all)
          # eval_status_loss.append(eval_loss_all)

          # if eval_acc_all > best_eval_accuracy:
          saver.save(sess, os.path.join(FLAGS.ckpt_dir, '{}_{}_{}_{}_{}_{}.ckpt'.format(
	                                      FLAGS.model, FLAGS.learning_rate,
	                                      FLAGS.batch_size, FLAGS.weight_decay,
	                                      FLAGS.dropout_keep_prob, step//40000)))

          #   best_eval_accuracy = eval_acc_all
          # print('best_eval_accuracy:%.5f'%best_eval_accuracy)

    except tf.errors.OutOfRangeError:
      saver.save(sess, os.path.join(FLAGS.ckpt_dir,
                                    '{}_{}_{}_{}_{}_final.ckpt'.format(
                                    FLAGS.model, FLAGS.learning_rate,
                                    FLAGS.batch_size, FLAGS.weight_decay,
                                    FLAGS.dropout_keep_prob
                                    )))
      print('Done training -- epoch limit reached')
      # eval_status['steps'] = eval_status_steps
      # eval_status['accuracy'] = eval_status_accuracy
      # eval_status['loss'] = eval_status_loss
      # eval_status.to_csv('./test_status.csv', index=False)  # save test accuracy and loss
    finally:
      coord.request_stop()
    coord.join(threads)
  # print(best_eval_accuracy)

if __name__ == '__main__':
  tf.app.run()
