import io
import argparse
import numpy as np
import time
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import LSTMCell, BasicRNNCell, GRUCell

from utils import get_logger
from data import DataProvider
from context_rnn import SCRNCell


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
    parser.add_argument('--cell', type=str, default='scrn', help='type of cell',
                        choices=['scrn', 'lstm', 'rnn', 'gru'])
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.95, help='alpha')
    parser.add_argument('--seq_length', type=int, default=100, help='seq length')
    parser.add_argument('--rnn_size', type=int, default=128, help='rnn size')
    parser.add_argument('--context_size', type=int, default=40, help='context size')
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
    context_size = args.context_size
    num_epochs = args.num_epoch
    learning_rate = args.lr
    alpha = args.alpha

    logger.info('build vocabulary')
    data_provider = DataProvider(train_text, seq_length, batch_size, logger)
    X_train, y_train = data_provider.get_data()

    logger.info('X.shape={}, y.shape={}'.format(X_train.shape, y_train.shape))

    vocab_size = data_provider.vocab_size

    input_data = tf.placeholder(tf.float32, [batch_size, seq_length, vocab_size])
    targets = tf.placeholder(tf.float32, [batch_size, vocab_size])

    test_data_provider = DataProvider(test_text, seq_length, batch_size, logger, data_provider.vocab)

    if args.cell == 'lstm':
        cell = LSTMCell(num_units=rnn_size)
    elif args.cell == 'rnn':
        cell = BasicRNNCell(num_units=rnn_size)
    elif args.cell == 'gru':
        cell = GRUCell(num_units=rnn_size)
    else:
        cell = SCRNCell(num_units=rnn_size, context_units=context_size, alpha=alpha)
    # initial_state = cell.zero_state(batch_size, tf.float32)

    # Define weights
    weights = {'out': tf.Variable(tf.random_normal([rnn_size, vocab_size]))}
    biases = {'out': tf.Variable(tf.random_normal([vocab_size]))}

    x = tf.unstack(input_data, seq_length, 1)
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)
    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    prediction = tf.nn.softmax(logits)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
    tf.summary.scalar('loss', loss_op)

    perplexity_op = tf.exp(loss_op)
    tf.summary.scalar('perplexity', perplexity_op)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(targets, 1))

    init = tf.global_variables_initializer()

    loss = 0
    merged = tf.summary.merge_all()

    start = time.time()

    with tf.Session() as sess:
        sess.run(init)

        train_writer = tf.summary.FileWriter('logs' + '/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs' + '/test')

        for e in range(num_epochs):

            summary = None
            run_metadata = None

            if e > 0 and e % 10 == 0:
                losses = []
                for _ in range(test_data_provider.num_batches):
                    X_test_batch, y_test_batch = test_data_provider.next_batch()
                    _loss = sess.run(loss_op, feed_dict={input_data: X_test_batch, targets: y_test_batch})
                    losses.append(_loss)
                perplexity = np.exp(np.mean(losses))
                summary = tf.Summary()
                summary.value.add(tag="perplexity", simple_value=perplexity)
                summary.value.add(tag="loss", simple_value=np.mean(losses))
                test_writer.add_summary(summary, e)

                test_data_provider.reset_batch_pointer()

                logger.info('Step {}, train time {}'.format(e, time.time() - start))

            else:
                for _ in range(data_provider.num_batches):
                    X_train_batch, y_train_batch = data_provider.next_batch()

                    run_metadata = tf.RunMetadata()

                    summary, _ = sess.run([merged, train_op],
                                          feed_dict={input_data: X_train_batch, targets: y_train_batch},
                                          run_metadata=run_metadata)

                train_writer.add_run_metadata(run_metadata, 'step%03d' % e)
                train_writer.add_summary(summary, e)

                data_provider.reset_batch_pointer()

        train_writer.close()
        test_writer.close()

        logger.info('Training time {}'.format(time.time() - start))