import io
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import LSTMCell

from utils import get_logger
from data import DataProvider
from context_rnn import SCRNNCell


def get_text():
    txt = ''
    with io.open('data/ptb/ptb.train.txt', encoding='utf-8') as f:
        txt += f.read().lower()
    return txt


def get_test_text():
    txt = ''
    with io.open('data/ptb/ptb.test.txt', encoding='utf-8') as f:
        txt += f.read().lower()
    return txt


if __name__ == '__main__':

    logger = get_logger()

    logger.info('read text')
    text = get_text()

    max_len = 100
    seq_length = max_len
    batch_size = 128
    rnn_size = 128
    num_epochs = 100
    learning_rate = 1e-1

    logger.info('build vocabulary')
    data_provider = DataProvider(text, max_len, batch_size, logger)
    X_train, y_train = data_provider.get_data()

    logger.info('X.shape={}, y.shape={}'.format(X_train.shape, y_train.shape))

    vocab_size = data_provider.vocab_size

    input_data = tf.placeholder(tf.float32, [batch_size, seq_length, vocab_size])
    targets = tf.placeholder(tf.float32, [batch_size, vocab_size])

    cell = LSTMCell(num_units=rnn_size)
    # cell = SCRNNCell(num_units=rnn_size, context_units=40, alpha=0.95)
    # initial_state = cell.zero_state(batch_size, tf.float32)

    # Define weights
    weights = {'out': tf.Variable(tf.random_normal([rnn_size, vocab_size]))}
    biases = {'out': tf.Variable(tf.random_normal([vocab_size]))}

    x = tf.unstack(input_data, max_len, 1)
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)
    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    prediction = tf.nn.softmax(logits)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(targets, 1))

    init = tf.global_variables_initializer()

    loss = 0

    with tf.Session() as sess:
        sess.run(init)
        for e in range(num_epochs):
            for _ in range(data_provider.num_batches):
                X_train_batch, y_train_batch = data_provider.next_batch()
                sess.run(train_op, feed_dict={input_data: X_train_batch, targets: y_train_batch})
                loss = sess.run(loss_op, feed_dict={input_data: X_train_batch, targets: y_train_batch})
            perplexity = np.exp(loss)
            logger.info("Step {}, Loss= {:.4f}, Perplexity= {:.3f}".format(e, loss, perplexity))
            data_provider.reset_batch_pointer()

        logger.info('test')
        test_text = get_test_text()
        test_data_provider = DataProvider(test_text, seq_length, batch_size, logger, data_provider.vocab)

        loss = []
        for _ in range(test_data_provider.num_batches):
            X_test_batch, y_test_batch = test_data_provider.next_batch()
            _loss = sess.run(loss_op, feed_dict={input_data: X_test_batch, targets: y_test_batch})
            loss.append(_loss)
        perplexity = np.exp(np.mean(loss))
        logger.info("Loss= {:.4f}, Perplexity= {:.3f}".format(np.mean(loss), perplexity))