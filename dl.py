import cPickle
import sys

import numpy as np
import tensorflow as tf
from keras.utils import np_utils

pickle_file = 'deduped_mnist.pickle'


def zero_center(X):
    X -= np.mean(X, axis=0)
    return X


def reformat(df, num_labels=10):
    dtype = 'float32'
    dataset, labels = df.iloc[:, :-1].values.astype(dtype), df.iloc[:, -1].values.astype(dtype)
    labels = np_utils.to_categorical(labels, num_labels)
    return dataset, labels


def accuracy(predictions, labels):
    """Calculates the accuracy of a model."""
    num_samples = predictions.shape[0]
    num_correct = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    return 100.0 * num_correct / num_samples


def load_data(subset=9999999):
    with open(pickle_file, 'rb') as f:
        save = cPickle.load(f)
        train = save['train_dataset']
        valid = save['valid_dataset']
        test = save['test_dataset']
        del save  # hint to help gc free up memory
        train_X, train_y = reformat(train.loc[:subset, :])
        valid_X, valid_y = reformat(valid.loc[:subset, :])
        test_X, test_y = reformat(test.loc[:subset, :])
    return train_X, train_y, valid_X, valid_y, test_X, test_y


def initialize_weight(shape, stddev=0.1):
    """Initializes weights variables for a layer"""
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def initialize_bias(shape, value=0.1):
    """Initializes bias variables for a layer"""
    return tf.Variable(tf.constant(value, shape=shape))


def grelu(f, alpha=0):
    """Generalized ReLU for vectors.

    Args:
        f: numeric vector
        alpha: 'leakiness' constant. 0 for canonical ReLU, 0.1 for 'leaky' ReLU

    Returns:
        max(alpha * f, f)
    """
    return tf.maximum(alpha * f, f)


def dense_layer(x, W, b):
    """Returns a fully-connected layer"""
    return grelu(tf.matmul(x, W) + b)


def conv_layer(x, W, b, stride, padding='SAME'):
    """Returns a convolution layer"""
    return grelu(tf.nn.conv2d(x, W, stride, padding=padding) + b)


def pool_layer(layer, ksize, stride, padding='VALID'):
    """Returns a pooling layer"""
    return tf.nn.max_pool(layer, ksize=ksize, strides=stride, padding=padding)


def convolve_and_pool_layer(x, W, b, conv_stride, pool_stride, ksize):
    """Convenience function for convolving and pooling layers"""
    layer = conv_layer(x, W, b, conv_stride)
    return pool_layer(layer, ksize, pool_stride)


def conv_to_dense_layer(layer, W, b):
    """Convenience function for transform from a conv/pool layer to a dense layer."""
    shape = layer.get_shape().as_list()
    reshape = tf.reshape(layer, [shape[0], shape[1] * shape[2] * shape[3]])
    return dense_layer(reshape, W, b)


def get_mini_batch(l, batch_size):
    """Generator for mini batches used in training."""
    for i in xrange(0, len(l), batch_size):
        yield l[i:i + batch_size]


def setup_convnet(
    x_train,
    y_train,
    x_valid,
    x_test,
    l2_weight,
    image_size=28,
    num_labels=10,
    p_keep=0.5
):
    patch_size = 5
    num_channels = 1

    # Variables.
    n_1 = 1024
    d_1 = 32
    d_2 = 64
    pool_step = 2
    pool_size = 2
    conv_stride = [1, 1, 1, 1]
    pool_stride = [1, 2, 2, 1]
    ksize = [1, 2, 2, 1]
    layer_1_dim = (image_size - pool_size) / pool_step + 1
    layer_2_dim = (layer_1_dim - pool_size) / pool_step + 1

    w_1 = initialize_weight([patch_size, patch_size, num_channels, d_1])
    b_1 = initialize_bias([d_1])

    w_2 = initialize_weight([patch_size, patch_size, d_1, d_2])
    b_2 = initialize_bias([d_2])

    w_3 = initialize_weight([layer_2_dim * layer_2_dim * d_2, n_1])
    b_3 = initialize_bias([n_1])

    w_o = initialize_weight([n_1, num_labels])
    b_o = initialize_bias([num_labels])

    # Training computation with dropout
    layer_1 = convolve_and_pool_layer(x_train, w_1, b_1, conv_stride, pool_stride, ksize)
    layer_2 = convolve_and_pool_layer(layer_1, w_2, b_2, conv_stride, pool_stride, ksize)
    layer_3 = tf.nn.dropout(conv_to_dense_layer(layer_2, w_3, b_3), p_keep)
    logits = tf.matmul(layer_3, w_o) + b_o

    data_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_train))
    reg_loss = (tf.nn.l2_loss(w_1) + tf.nn.l2_loss(w_2) + tf.nn.l2_loss(w_3) + tf.nn.l2_loss(w_o))
    loss = data_loss + l2_weight * reg_loss

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

    layer_1 = convolve_and_pool_layer(x_valid, w_1, b_1, conv_stride, pool_stride, ksize)
    layer_2 = convolve_and_pool_layer(layer_1, w_2, b_2, conv_stride, pool_stride, ksize)
    layer_3 = conv_to_dense_layer(layer_2, w_3, b_3)
    valid_prediction = tf.nn.softmax(tf.matmul(layer_3, w_o) + b_o)

    layer_1 = convolve_and_pool_layer(x_test, w_1, b_1, conv_stride, pool_stride, ksize)
    layer_2 = convolve_and_pool_layer(layer_1, w_2, b_2, conv_stride, pool_stride, ksize)
    layer_3 = conv_to_dense_layer(layer_2, w_3, b_3)
    test_prediction = tf.nn.softmax(tf.matmul(layer_3, w_o) + b_o)
    return train_prediction, test_prediction, valid_prediction, loss


def main():
    num_labels = 10
    image_size = 28
    l2_weight = 5e-4
    batch_size = 128
    num_epochs = 2
    learning_rate = 0.001
    train_X, train_y, valid_X, valid_y, test_X, test_y = load_data()
    resize_shape = (-1, image_size, image_size, 1)
    train_X = train_X.reshape(resize_shape)
    valid_X = zero_center(valid_X.reshape(resize_shape))
    test_X = zero_center(test_X.reshape(resize_shape))
    graph = tf.Graph()
    with graph.as_default():
        tf_train_X = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 1))
        tf_train_y = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_X = tf.constant(valid_X)
        tf_test_X = tf.constant(test_X)
        train_prediction, test_prediction, valid_prediction, total_loss = setup_convnet(
            tf_train_X, tf_train_y, tf_valid_X, tf_test_X, l2_weight
        )
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    idx = range(len(train_X))
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for epoch in xrange(num_epochs):
            np.random.shuffle(idx)
            batches_X = get_mini_batch(train_X[idx, :], batch_size)
            batches_y = get_mini_batch(train_y[idx], batch_size)
            wat = 0
            for batch_data, batch_labels in zip(batches_X, batches_y):
                batch_data = zero_center(batch_data)
                wat += 1
                if len(batch_data) != batch_size:
                    continue
                feed_dict = {tf_train_X: batch_data, tf_train_y: batch_labels}
                _, l, predictions = session.run(
                    [optimizer, total_loss, train_prediction],
                    feed_dict=feed_dict
                )
                if wat % 500 == 0:
                    print '-' * 100
                    print('Doing step {} of epoch {}'.format(wat, epoch))
                    print "Minibatch loss:", l
                    print "Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)
                    print "Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_y)
        print "Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_y)


if __name__ == '__main__':
    # For reproducibility
    np.random.seed(1337)
    sys.exit(main('poolconvnet'))
