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
    W_reshaped = tf.reshape(tf.transpose(W), [-1, 28, 28, 1])
    W_summary = tf.summary.image('W_filters', W_reshaped, max_outputs=10)

    # the model
    y = tf.matmul(x, W) + b

    # loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    tf.summary.scalar('xent', cross_entropy)

    # training
    trainer = tf.train.GradientDescentOptimizer(0.2)
    train_step = trainer.minimize(cross_entropy)

    # testing
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    tf.summary.scalar('accuracy', accuracy)
    
    # summary all
    summary = tf.summary.merge_all()

    # saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # init Vars
        writer = tf.summary.FileWriter("LogDir") # summary writer
        writer.add_graph(sess.graph)

        for i in range(500):
            batchX, batchY = mnist.train.next_batch(250)
            _, sum = sess.run([train_step, summary], feed_dict={x:batchX, y_:batchY})
            if (i+1) % 100 == 0:
                saver.save(sess, './mnist-logit/model', global_step=100, write_meta_graph=True)
            writer.add_summary(sum, i)


if __name__ == '__main__':
    import os, shutil
    if os.path.exists('LogDir'):
        shutil.rmtree('LogDir')
    main( )
