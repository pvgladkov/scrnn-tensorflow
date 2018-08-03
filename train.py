import io
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import LSTMCell

from utils import get_logger
from data import DataProvider
from context_rnn import SCRNNCell


def get_text():
    t_txt = ''
    with io.open('data/ptb/ptb.train.txt', encoding='utf-8') as f:
        t_txt += f.read().lower()
    test_txt = ''
    with io.open('data/ptb/ptb.test.txt', encoding='utf-8') as f:
        test_txt += f.read().lower()
    return t_txt, test_txt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', type=str, default='scrnn', help='type of cell', choices=['scrnn', 'lstm'])
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--seq_length', type=int, default=100, help='seq length')
    parser.add_argument('--rnn_size', type=int, default=128, help='rnn size')
    parser.add_argument('--num_epoch', type=int, default=100, help='num epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    return parser.parse_args()


if __name__ == '__main__':

    logger = get_logger()

    logger.info('read text')
    train_text, test_text = get_text()

    args = parse_args()

    seq_length = args.seq_length
    batch_size = args.batch_size
    rnn_size = args.rnn_size
    num_epochs = args.num_epoch
    learning_rate = args.lr

    logger.info('build vocabulary')
    data_provider = DataProvider(train_text, seq_length, batch_size, logger)
    X_train, y_train = data_provider.get_data()

    logger.info('X.shape={}, y.shape={}'.format(X_train.shape, y_train.shape))

    vocab_size = data_provider.vocab_size

    input_data = tf.placeholder(tf.float32, [batch_size, seq_length, vocab_size])
    targets = tf.placeholder(tf.float32, [batch_size, vocab_size])

    if args.cell == 'lstm':
        cell = LSTMCell(num_units=rnn_size)
    else:
        cell = SCRNNCell(num_units=rnn_size, context_units=40, alpha=0.95)
    # initial_state = cell.zero_state(batch_size, tf.float32)

    # Define weights
    weights = {'out': tf.Variable(tf.random_normal([rnn_size, vocab_size]))}
    biases = {'out': tf.Variable(tf.random_normal([vocab_size]))}

    x = tf.unstack(input_data, seq_length, 1)
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
            logger.info("Step {}, Loss={:.4f}, Perplexity={:.3f}".format(e, loss, perplexity))
            data_provider.reset_batch_pointer()

        logger.info('test')
        test_data_provider = DataProvider(test_text, seq_length, batch_size, logger, data_provider.vocab)

        loss = []
        for _ in range(test_data_provider.num_batches):
            X_test_batch, y_test_batch = test_data_provider.next_batch()
            _loss = sess.run(loss_op, feed_dict={input_data: X_test_batch, targets: y_test_batch})
            loss.append(_loss)
        perplexity = np.exp(np.mean(loss))
        logger.info("Loss={:.4f}, Perplexity={:.3f}".format(np.mean(loss), perplexity))