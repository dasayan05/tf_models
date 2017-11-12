import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.rnn import BasicLSTMCell, static_rnn

from numpy import zeros

BATCH_SIZE = 100
INP_LEN, HID_LEN, OUT_LEN = 28, 20, 10
SEQ_LEN = 28
TOTAL_EPOCH = 1000

h = 1
LOG_DIR = "lstm_mnist_Log/"

def rnn(x: 'I/P tensor for one timestep', h_prev: 'previous hidden state'):
    with tf.name_scope('hidden'):
        state = None
        with tf.variable_scope('param', reuse=tf.AUTO_REUSE):
            Wxh = tf.get_variable('Wxh', shape=(INP_LEN, HID_LEN), dtype=tf.float64, 
                initializer=tf.initializers.random_normal)
            Whh = tf.get_variable('Whh', shape=(HID_LEN, HID_LEN), dtype=tf.float64,
                initializer=tf.initializers.random_normal)
            bh = tf.get_variable('bh', shape=(HID_LEN,), dtype=tf.float64, 
                initializer=tf.initializers.zeros)
        global h
        state = tf.tanh(tf.matmul(x, Wxh) + tf.matmul(h_prev, Whh) + bh, name='h'+str(h))
        h += 1
        return state

def main():

    # data collection
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.name_scope('inputs'):
        X_ = tf.placeholder(tf.float64, shape=(None,SEQ_LEN,INP_LEN), name='X_')
        X = tf.unstack(X_, axis=1)
    #state = tf.placeholder(tf.float64, shape=(None,HID_LEN), name='h0')

    #for ip in X:
    #    state = rnn(ip, state)
    lstm = BasicLSTMCell(HID_LEN, forget_bias=1.0)
    outputs, _ = static_rnn(lstm, X, dtype=tf.float64)

    with tf.name_scope('output'):
        Y_ = tf.placeholder(tf.float64, shape=(None,10), name='Y_')
        with tf.variable_scope('param', reuse=tf.AUTO_REUSE):
            Who = tf.get_variable('Who', shape=(HID_LEN,OUT_LEN), dtype=tf.float64,
                initializer=tf.initializers.random_normal)
            bo = tf.get_variable('bo', shape=(OUT_LEN,), dtype=tf.float64)
        logit = tf.add(tf.matmul(outputs[-1], Who), bo, name='logit')

    # predict/testing tensors
    with tf.name_scope('prediction'):
        correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    with tf.name_scope('training'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y_), name='loss')
        optimizer = tf.train.RMSPropOptimizer(0.1)
        train_step = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(LOG_DIR)
        writer.add_graph(sess.graph)

        for E in range(TOTAL_EPOCH):

            epoch_avg_loss = 0

            for B in range(int(mnist.train.num_examples/BATCH_SIZE)):
                Im, Lb = mnist.train.next_batch(BATCH_SIZE)
                Im = Im.reshape(BATCH_SIZE, 28, 28).swapaxes(1,2)
                _, l_, acc_ = sess.run([train_step, loss, accuracy],
                    feed_dict={
                        X_:Im,
                        Y_:Lb,
                        #'h0:0': zeros((BATCH_SIZE,HID_LEN))
                    })
                #if B % 250 == 0:
                #    print('BatchIter:{0}'.format(B))

                epoch_avg_loss += l_
            epoch_avg_loss /= mnist.train.num_examples/BATCH_SIZE

            I_, L_ = mnist.validation.images, mnist.validation.labels
            I_ = I_.reshape(5000, 28, 28).swapaxes(1,2)
            test_acc = sess.run(accuracy, feed_dict={X_:I_,Y_:L_})

            if E % 10 == 0:
                print('Epoch:{0},AVGLoss:{1},Acc:{2}'.format(E,epoch_avg_loss,acc_))
                print('TestAcc:{0}'.format(test_acc))

if __name__ == '__main__':
    import os, shutil
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    main()