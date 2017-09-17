from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import zipfile
import content
from config import *

flags = tf.flags
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("data_file", None, "Input zipfile")
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_integer("num_words", 20, "Number of words to generate")
flags.DEFINE_string("text", "", "Text to start evalutating from")
FLAGS = flags.FLAGS

def read_data(zf):
  """
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary, words)
  """
  data = content.read_zip_content(zf)

  n_train_samples = int(len(data) * 8 / 10)
  start_valid_samples = int(len(data) * 9 / 10)

  word_to_id, words = content.build_vocabulary(data)

  train_data = content.content_to_word_ids(data[:n_train_samples], word_to_id)
  valid_data = content.content_to_word_ids(data[n_train_samples:start_valid_samples], word_to_id)
  test_data = content.content_to_word_ids(data[start_valid_samples:], word_to_id)

  return train_data, valid_data, test_data, word_to_id, words

class RNN:
  def __init__(self, inputs, config):
    size = config.hidden_size
    vocab_size = config.vocab_size
    inputs = tf.reshape(inputs, [-1, config.num_steps])
    embedding = tf.get_variable(
        "embedding", [vocab_size, size], dtype=tf.float32)

    inputs = tf.nn.embedding_lookup(embedding, inputs)

    cell = tf.contrib.rnn.MultiRNNCell(
      [tf.contrib.rnn.BasicLSTMCell(size) for _ in range(config.num_layers)])

    self.state = self.initial_state = cell.zero_state(config.batch_size, tf.float32)
    #if config.is_training and config.keep_prob < 1:
    #  inputs = tf.nn.dropout(inputs, config.keep_prob)

    #print("inputs {}".format(inputs.shape))
    inputs = tf.unstack(inputs, num=config.num_steps, axis=1)
    #print("inputs {}".format(inputs[0].shape))

    outputs, next_state = tf.nn.static_rnn(cell, inputs, self.state, dtype = tf.float32)
    #print ("state {}".format(state.shape))
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)

    logits = []
    for output in outputs:
      logits.append(tf.nn.xw_plus_b(output, softmax_w, softmax_b))

    logits = tf.reshape(tf.concat(logits, 1), [config.batch_size, config.num_steps, config.vocab_size])

    self.logits = logits
    self.next_state = next_state

class Trainer:
  def __init__(self, train_data, config):
    self.config = config
    self.learning_rate = tf.Variable(1.0, trainable=False)

    inputs, targets = content.text_producer(train_data, config.batch_size, config.num_steps)

    self.epoch_size = ((len(train_data) // config.batch_size) - 1) // config.num_steps
    print ("epoch_size = {}".format(self.epoch_size))
    onehot_y = tf.reshape(tf.one_hot(targets, config.vocab_size), [config.batch_size, config.num_steps, config.vocab_size])

    self.rnn = RNN(inputs, config)

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        self.rnn.logits,
        targets,
        tf.ones([config.batch_size, config.num_steps], dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True)

    self.cost = tf.reduce_sum(loss)
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)

    self.training = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

class Runner:
  def __init__(self, config):
    self.inputs = tf.placeholder(tf.int32)
    self.rnn = RNN(self.inputs, config)

    logits = tf.reshape(self.rnn.logits, [config.vocab_size])

    self.output = tf.argmax(logits)
    self._feed_dict = {}

  def next(self, session, word_id):
    self._feed_dict[self.inputs] = word_id
    output, state = session.run([self.output, self.rnn.next_state], feed_dict = self._feed_dict)
    self._feed_dict[self.rnn.state[0]] = state[0]
    self._feed_dict[self.rnn.state[1]] = state[1]
    return output

def main(_):
  zf = zipfile.ZipFile(FLAGS.data_file)
  train_data, valid_data, test_data, word_to_id, words = read_data(zf)

  config = Config(FLAGS.train)

  if FLAGS.train:
    with tf.Graph().as_default():
      init = tf.global_variables_initializer()

      trainer = Trainer(train_data, config)

      sv = tf.train.Supervisor(logdir=FLAGS.save_path)

      with sv.managed_session() as session:
        session.run(init)

        feed_dict = {}

        for epoch in range(config.max_max_epoch):
          feed_dict[trainer.learning_rate] = config.learning_rate * (config.lr_decay ** max(epoch + 1 - config.max_epoch, 0.0))
          for i in range(trainer.epoch_size):
            _, the_cost, current_state = session.run([trainer.training, trainer.cost, trainer.rnn.next_state],
                                                     feed_dict = feed_dict)
            feed_dict[trainer.rnn.state[0]] = current_state[0]
            feed_dict[trainer.rnn.state[1]] = current_state[1]

            if i % 10 == 0:
              print("%d: %d, cost = %f" % (epoch, i, the_cost))

          print ("%d: cost = %f" % (epoch, the_cost))
          if FLAGS.save_path:
            print("Saving model to %s." % FLAGS.save_path)
            sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)
  else:
    if len(FLAGS.text) == 0:
      text = content.split_text("Hur var det i ")
    else:
      text = content.split_text(FLAGS.text)

    text_ids = [word_to_id[word] for word in text]

    with tf.Graph().as_default():

      runner = Runner(config)

      sv = tf.train.Supervisor(logdir=FLAGS.save_path, save_model_secs=0, save_summaries_secs=0)
      with sv.managed_session() as session:
        for i in range(len(text_ids)):
          output = runner.next(session, text_ids[i])

        text_ids.append(output)
        for i in range(FLAGS.num_words):
          output = runner.next(session, output)
          text_ids.append(output)

          #print (words)
          #print (text_ids)
        print (content.to_text(text_ids, words))

if __name__ == "__main__":
  tf.app.run()
