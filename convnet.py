import tensorflow as tf   # regular tf import
from tensorflow.examples.tutorials.mnist import input_data
from mnist_mlp import MLP # for the fully connected part
import os, shutil

class ConvPool(object):
    'Layer representing a conv-layer followed by a pooling'

    # global instance counter
    __ConvPools = 0

    def __init__(self, kernel_size=(7,7), in_ch=1, out_ch=32, pool_size=(2,2), reuse=True):
        'the class initializer'

        # track counter global/local
        ConvPool.__ConvPools += 1
        self.id = ConvPool.__ConvPools
        tf.logging.info('ConvPool {0} created'.format(self.id))

        # args validation
        if len(kernel_size) != 2:
            raise 'Kernel should be a 2-tuple'

        # track architecture parameters
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.pool_size = pool_size

        # create weight
        if reuse:
            self.__init_weights__(re_use=tf.AUTO_REUSE)
        else:
            self.__init_weights__()

    def __init_weights__(self, re_use=None):
        'private function for weight initialization'

        var_scope = 'ConvPool' + str(self.id) + '_vars'

        with tf.variable_scope(var_scope, reuse=re_use):

            # the kernel/filter/conv parameters
            self.W = tf.get_variable(var_scope + 'W', shape=(*self.kernel_size, self.in_ch, self.out_ch),
                dtype=tf.float32, initializer=tf.initializers.random_normal)
            self.b = tf.get_variable(var_scope + 'b', shape=(self.out_ch), dtype=tf.float32,
                initializer=tf.initializers.zeros)

    def make_model(self, x: '[B,H,W,C]', scope_name='model'):
        'construct the model'

        with tf.name_scope(scope_name):
            self.h = tf.nn.relu(tf.nn.conv2d(x, self.W, strides=[1,1,1,1], padding='SAME') + self.b)
            self.hp = tf.nn.max_pool(self.h, ksize=[1,*self.pool_size, 1],
                strides=[1,*self.pool_size, 1], padding='SAME')

        return self.hp

def main( args = None ):

    # load MNIST dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.name_scope('phs'):
        x = tf.placeholder(tf.float32, shape=(None,28,28,1), name='x')
        y = tf.placeholder(tf.float32, shape=(None,10), name='y')

    # First Conv+Pool
    convpool1 = ConvPool((7,7), 1, 32, (2,2))
    cp1 = convpool1.make_model(x, scope_name='cp1')

    # Second Conv+Pool
    convpool2 = ConvPool((5,5), 32, 64, (2,2))
    cp2 = convpool2.make_model(cp1, scope_name='cp2')

    # flatten
    cp2_f = tf.reshape(cp2, shape=(-1, 7*7*64), name='flatten')

    # the fully connected layer
    mlp = MLP([7*7*64, 1024, 10], dtype=tf.float32)
    y_ = mlp.make_model(cp2_f, scope_name='FC')

    # the loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y), name='loss')

    # optimizer
    optimizer = tf.train.AdamOptimizer(tf.app.flags.FLAGS.lr)
    with tf.name_scope('optim'):
        train_step = optimizer.minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(tf.app.flags.FLAGS.logdir)
        writer.add_graph(graph=sess.graph)

        for I in range(tf.app.flags.FLAGS.epoch):
            X, Y = mnist.train.next_batch(tf.app.flags.FLAGS.batch_size)
            _, l = sess.run([train_step, loss], feed_dict={x: X.reshape((-1,28,28,1)), y: Y})

            if I % 10 == 0:
                print('Loss {0}'.format(l))

if __name__ == '__main__':
    # For running the script as main()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.logging.info('Running {0}'.format(__file__))

    tf.app.flags.DEFINE_float('lr', 0.01, 'Learning rate')
    tf.app.flags.DEFINE_string('logdir', 'MNIST_CNNLog', 'logging directory')
    tf.app.flags.DEFINE_integer('epoch', 10000, 'Epochs')
    tf.app.flags.DEFINE_integer('batch_size', 500, 'Minibatch size')

    if os.path.exists(tf.app.flags.FLAGS.logdir):
        shutil.rmtree(tf.app.flags.FLAGS.logdir)

    tf.app.run()

if __name__ == os.path.splitext(os.path.basename(__file__))[0]:
    # for importing from this file
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.logging.info('Importing from {0}'.format(os.path.basename(__file__)))