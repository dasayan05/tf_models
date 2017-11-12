import os, shutil
import tensorflow as tf
from get_data import get_iris_data
from numpy.random import randn
from numpy import zeros

LOG_DIR = "IRIS_MLP_Log"
n_hidden = 3
with_relu = True

def main():
    # get the IRIS dataset in proper form
    irisX, irisY, info = get_iris_data(return_X_y=True, one_hot=True)

    x = tf.placeholder(tf.float64, shape=(None, info['dim']), name='x')
    with tf.name_scope('params'):
        with tf.name_scope('hidden1'):
            W1 = tf.Variable(randn(info['dim'], n_hidden), name='W1', dtype=tf.float64)
            b1 = tf.Variable(zeros(n_hidden,), name='b1', dtype=tf.float64)
        with tf.name_scope('hidden2'):
            W2 = tf.Variable(randn(n_hidden, info['n_class']), name='W2', dtype=tf.float64)
            b2 = tf.Variable(zeros(info['n_class'],), name='b2', dtype=tf.float64)

    y_ = tf.placeholder(tf.float64, shape=(None, info['n_class']), name='y_')

    # the model
    if with_relu:
        h1_act = tf.nn.relu( tf.matmul(x, W1) + b1 )
    else:
        h1_act = tf.matmul(x, W1) + b1
    h1_act_0_summary = tf.summary.histogram('h1_act_hist_0', h1_act[:,0])
    h1_act_1_summary = tf.summary.histogram('h1_act_hist_1', h1_act[:,1])
    h1_act_2_summary = tf.summary.histogram('h1_act_hist_2', h1_act[:,2])

    h2_act = tf.matmul(h1_act, W2) + b2

    # predict/testing tensors
    with tf.name_scope('prediction'):
        correct_prediction = tf.equal(tf.argmax(h2_act, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    accuracy_summary = tf.summary.scalar('accuracy_train', accuracy)

    # loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h2_act, labels=y_))
    loss_summary = tf.summary.scalar('loss_value', loss)

    with tf.name_scope('training'):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    all_summary = tf.summary.merge([loss_summary, accuracy_summary,
        h1_act_0_summary,
        h1_act_1_summary,
        h1_act_2_summary])

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(LOG_DIR)
        
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        for I in range(1000):
            summar_, _ = sess.run([all_summary, train_step], feed_dict={x: irisX, y_:irisY})
            writer.add_summary(summar_, I)

if __name__ == '__main__':
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    main()