"""
Author: Ayan Das
Simple linear model for MNIST classification
"""

import tensorflow as tf
from tf_extra import tf_placeholders, tf_Variables, float_default_prec
from tensorflow.examples.tutorials.mnist import input_data

def main( ):
    """ The main function for testing """

    # load the dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # the placeholders 'x' and 'y_'
    x, y_ = tf_placeholders(['x', 'y_'], [(None, 784), (None, 10)])

    # the variables 'W' and 'b'
    W_init = tf.zeros((784, 10), dtype=float_default_prec)
    b_init = tf.zeros((10,), dtype=float_default_prec)
    W, b = tf_Variables(['W', 'b'], [W_init, b_init])

    # the model
    y = tf.matmul(x, W) + b

    # loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    # training
    trainer = tf.train.GradientDescentOptimizer(0.2)
    train_step = trainer.minimize(cross_entropy)

    # testing
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # init Vars

        for _ in range(1000):
            batchX, batchY = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x:batchX, y_:batchY})

            # test on test dataset
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    main( )
