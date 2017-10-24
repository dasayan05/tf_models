"""
Author: Ayan Das
Checking a very simple linear model
"""

from numpy import float_ as float_default_prec
import tensorflow as tf
from tf_extra import tf_placeholders, tf_Variables


def main():
    """ The main function for testing """

    # placeholder for input data & target
    x, y = tf_placeholders(['x', 'y'], None)

    # the model parameters
    W, b = tf_Variables(['W', 'b'], [[-1.], [3.]])

    # the model
    y_ = x * W + b

    # loss calculation
    sq_err = tf.square(y - y_)
    loss = tf.reduce_sum(sq_err)

    # optimizer creation
    optim = tf.train.GradientDescentOptimizer(0.02)
    train_step = optim.minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for indx in range(1000):
            lval, _, w_val, b_val = sess.run([loss, train_step, W, b], feed_dict={x:[1,2,3,4], y:[2,4,6,8]})
            print('Iteration:{}, Loss:{}, W:{}, b:{}'.format(indx,lval,w_val,b_val))

if __name__ == '__main__':
    main( )
