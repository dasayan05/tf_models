import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os, shutil

class MLP(object):
    'A simple multi-layer perceptron'

    # global instance counter
    __MLPs = 0

    def __init__(self, layer_dims: '3-tuple', reuse=True):
        'the class initializer'

        # handle global and local counting
        MLP.__MLPs += 1
        self.id = MLP.__MLPs
        tf.logging.info('MLP {0} created'.format(self.id))
        tf.logging.info('MLP architecture is {0}-{1}-{2}'.format(*tuple(layer_dims)))

        if len(layer_dims) != 3:
            raise 'layer_dims should be a 3-tuple'

        # track the architecture
        self.layer_dims = layer_dims

        # create weights
        if reuse:
            self.__init_weights__(re_use=tf.AUTO_REUSE)
        else:
            self.__init_weights__()

    def __init_weights__(self, re_use=None):
        'private function for weight initialization'

        var_scope = 'MLP' + str(self.id) + '_vars'

        with tf.variable_scope(var_scope, reuse=re_use):
            
            # first layer parameters
            self.W1 = tf.get_variable(var_scope+'_W1', shape=(self.layer_dims[0], self.layer_dims[1]), 
                dtype=tf.float64, initializer=tf.initializers.random_normal)
            self.b1 = tf.get_variable(var_scope+'_b1', shape=(self.layer_dims[1],), dtype=tf.float64,
                initializer=tf.initializers.zeros)

            # second layer parameters
            self.W2 = tf.get_variable(var_scope+'_W2', shape=(self.layer_dims[1], self.layer_dims[2]),
                dtype=tf.float64, initializer=tf.initializers.random_normal)
            self.b2 = tf.get_variable(var_scope+'_b2', shape=(self.layer_dims[2],), dtype=tf.float64,
                initializer=tf.initializers.zeros)

    def W_filter_summary(self):
        'summary for visualizing W1 weights as filters'

        W1_sum_name = 'MLP' + str(self.id) + '_W1_summary'

        with tf.name_scope('weight_filter'):
            W1_reshaped = tf.reshape(tf.transpose(self.W1), [-1, 28, 28, 1])
            W_summary = tf.summary.image(W1_sum_name, W1_reshaped, max_outputs=20)

        return W_summary

    def make_model(self, x: 'batch tensor', scope_name='model'):
        'create the MLP and return y(logits)'

        # The two-layer model
        with tf.name_scope(scope_name):
            # layer1: linear + relu
            self.h = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
            # layer2: linear
            self.y_ = tf.matmul(self.h, self.W2) + self.b2

        tf.logging.debug('Model {0} constructed'.format(self.id))
        return self.y_

    def get_accuracy(self, y, scope_name='prediction', with_summary=False):
        'get the accuracy tensor'

        # make sure to invoke self.make_model() first
        if hasattr(self, 'y_'):
            tf.logging.error('Model not completed')

        # for accuracy computation
        with tf.name_scope(scope_name):
            self.correct_prediction = tf.equal(tf.argmax(self.y_, 1), tf.argmax(y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float64))
            self.acc_summary = tf.summary.scalar('accuracy', self.accuracy)

        if with_summary:
            return self.accuracy, self.acc_summary
        else:
            return self.accuracy

def main( args = None ):

    mlp = MLP([784,tf.app.flags.FLAGS.hidden_size,10])
    w_summary = mlp.W_filter_summary()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.name_scope('phs'):
        x = tf.placeholder(tf.float64, shape=(None,784), name='x')
        y = tf.placeholder(tf.float64, shape=(None,10), name='y')

    y_ = mlp.make_model(x, scope_name='model')
    acc, acc_summary = mlp.get_accuracy(y, scope_name='prediction', with_summary=True)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
        loss_summary = tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(tf.app.flags.FLAGS.lr)
    with tf.name_scope('optim'):
        train_step = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer_train = tf.summary.FileWriter(tf.app.flags.FLAGS.logdir + '/' + 'train')
        tf.logging.info('train logs into {0}'.format(tf.app.flags.FLAGS.logdir + '/' + 'train'))
        writer_test = tf.summary.FileWriter(tf.app.flags.FLAGS.logdir + '/' + 'test')
        tf.logging.info('test logs into {0}'.format(tf.app.flags.FLAGS.logdir + '/' + 'test'))

        writer_train.add_graph(sess.graph)
        writer_test.add_graph(sess.graph)

        for I in range(tf.app.flags.FLAGS.epoch):
            X, Y = mnist.train.next_batch(tf.app.flags.FLAGS.batch_size)
            _, l_, l_sum, acc_sum = sess.run([train_step, loss, loss_summary, acc_summary], feed_dict={x:X, y:Y})
            if I % 20 == 0:
                tf.logging.debug('epoch {0}, train-loss {1}'.format(I, l_))
                writer_train.add_summary(l_sum, global_step=I)
                writer_train.add_summary(acc_sum, global_step=I)

            if I % 200 == 0:
                X, Y = mnist.test.images, mnist.test.labels
                acc_, acc_sum, l_sum, w_sum = sess.run([acc, acc_summary, loss_summary, w_summary],
                    feed_dict={x: X, y: Y})
                tf.logging.debug('test-accuracy {0}'.format(acc_))
                writer_test.add_summary(acc_sum, global_step=I)
                writer_test.add_summary(l_sum, global_step=I)

                writer_train.add_summary(w_sum, global_step=I)

if __name__ == '__main__':
    # For running the script as main()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.logging.info('Running {0}'.format(__file__))

    tf.app.flags.DEFINE_float('lr', 0.01, 'Learning rate')
    tf.app.flags.DEFINE_string('logdir', 'MNIST_MLPLog', 'logging directory')
    tf.app.flags.DEFINE_integer('hidden_size', 300, 'Number of neurons in hidden layer')
    tf.app.flags.DEFINE_integer('epoch', 10000, 'Epochs')
    tf.app.flags.DEFINE_integer('batch_size', 500, 'Minibatch size')

    if os.path.exists(tf.app.flags.FLAGS.logdir):
        shutil.rmtree(tf.app.flags.FLAGS.logdir)

    tf.app.run()

if __name__ == os.path.splitext(os.path.basename(__file__))[0]:
    # for importing from this file
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.logging.info('Importing from {0}'.format(__file__))