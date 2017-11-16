"""
Implementation of Capsule Network
https://arxiv.org/pdf/1710.09829.pdf
Author: Ayan Das
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os, shutil
from numpy import zeros, float32

bsize = 4 # BATCH SIZE. 128 maybe or 256
epochs = 100
MPlus = 0.9
MMinus = 0.1
lam = 0.5
eps = 1e-15
reco_loss_importance = 5e-4
routing_iter = 4

def squash(x, axis):
    vec_squared_norm = tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + eps)
    return tf.multiply(scalar_factor, x)

def main( args=None ):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    conv1_kernel_size = (9,9)
    conv1_fea_maps = 256
    conv1_strides = [1,1,1,1]

    convcaps_kernel_size = (9,9)
    convcaps_capsule_dim = 8
    convcaps_fea_maps = 32
    convcaps_stride = [1,2,2,1]

    n_class = 10
    out_capsule_dim = 16

    with tf.name_scope('phs') as phs:
        x = tf.placeholder(tf.float32, shape=(None,28,28,1), name='x')
        y = tf.placeholder(tf.float32, shape=(None,10), name='y')

    with tf.variable_scope('params') as params:
        with tf.variable_scope('Inp2Conv'):
            W_conv1 = tf.get_variable('W_conv1', shape=(*conv1_kernel_size, 1, conv1_fea_maps),
                dtype=tf.float32, initializer=tf.initializers.random_normal)
            b_conv1 = tf.get_variable('b_conv1', shape=(1,1,1,conv1_fea_maps), dtype=tf.float32,
                initializer=tf.initializers.random_uniform)

    with tf.name_scope('conv1'):
        conv1_pre = tf.nn.conv2d(x, W_conv1, strides=conv1_strides, padding='VALID') + b_conv1
        conv1_act = tf.nn.relu(conv1_pre, name='conv1_act')

    with tf.variable_scope(params):
        with tf.variable_scope('Conv2Caps'):
            w_var_name = 'Conv2Caps' + '_W'
            b_var_name = 'Conv2Caps' + '_b'

            W_convcaps = tf.get_variable(w_var_name, dtype=tf.float32,
                shape=(*convcaps_kernel_size, conv1_fea_maps, convcaps_capsule_dim*convcaps_fea_maps),
                initializer=tf.initializers.random_normal)
            b_convcaps = tf.get_variable(b_var_name, dtype=tf.float32,
                shape=(1,1,1,convcaps_capsule_dim*convcaps_fea_maps),
                initializer=tf.initializers.zeros)

    with tf.name_scope('PrimCaps'):
        primecaps_act = tf.nn.relu(
                tf.nn.conv2d(conv1_act, W_convcaps, convcaps_stride, 'VALID') + b_convcaps
            ) # (B x 6 x 6 x 256)

        primecaps_act = tf.split(primecaps_act, axis=3, num_or_size_splits=convcaps_fea_maps,
            name='primecaps_act') # [(B x 6 x 6 x 8) * 32]

    with tf.name_scope('CapsFlatten'):
        primecaps_cap_r = [
            tf.reshape(act, shape=(-1, 6*6, convcaps_capsule_dim))
            for act in primecaps_act
        ] # [(B x 36 x 8) * 32]

        primecaps_cap_f_ = tf.concat(primecaps_cap_r, axis=1) # (B x 36*32 x 8) == (B x 1152 x 8)

        with tf.name_scope('PrimeCaps_squash'):
            # mathamatically stable
            primecaps_cap_f = tf.reshape(squash(primecaps_cap_f_, axis=2), 
                shape=(-1, 6*6*convcaps_fea_maps, 1, convcaps_capsule_dim)) # (B x 1152 x 1 x 8)

    with tf.variable_scope(params):
        with tf.variable_scope('Caps2Caps'):
            W = tf.get_variable('W', dtype=tf.float32, initializer=tf.initializers.random_normal,
                shape=(1, 6*6*convcaps_fea_maps, convcaps_capsule_dim, out_capsule_dim, n_class))
            # (1, 1152, 8, 16, 10)

    with tf.name_scope('Caps2Digs'):
        uj = []
        with tf.name_scope('pred_vec'):
            for c in range(n_class):
                uj.append( tf.matmul(primecaps_cap_f, tf.tile(W[...,c], [bsize,1,1,1]),
                    name='u_'+str(c)) ) # (B x 1152 x 1 x 16)

            # full 'prediction vector'; need it in 'routing'; (B x 1152 x 16 x 10)
            u = tf.squeeze(tf.stack(uj, axis=4), axis=None, name='u')

        bij = tf.constant(zeros((bsize, 6*6*convcaps_fea_maps, n_class), dtype=float32),
            dtype=tf.float32, name='bij') # (B x 1152 x 10)

        for route_iter in range(routing_iter):
            with tf.name_scope('route_' + str(route_iter)):
                cij = tf.nn.softmax(bij, dim=2) # (B x 1152 x 10)

                s = tf.reduce_sum(u * tf.reshape(cij, shape=(bsize,6*6*convcaps_fea_maps,1,n_class)),
                    axis=1, keep_dims=False) # (B x 16 x 10)

                v = squash(s, axis=1) # (B x 16 x 10)

                if route_iter < routing_iter - 1: # bij comp not required at the end
                    # Here comes 'routing'
                    v_r = tf.reshape(v, shape=(-1, 1, out_capsule_dim, n_class)) # (B x 1 x 16 x 10)
                    uv_dot = tf.reduce_sum(u*v_r, axis=2, name='uv')

                    bij += uv_dot

    with tf.name_scope('loss'):

        with tf.name_scope('recon_loss'):
            with tf.variable_scope(params):
                layer1 = tf.layers.Dense(512, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.initializers.random_normal)
                layer2 = tf.layers.Dense(1024, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.initializers.random_normal)
                layer3 = tf.layers.Dense(784, activation=tf.nn.sigmoid, use_bias=True, kernel_initializer=tf.initializers.random_normal)

            v_masked = tf.multiply(v, tf.reshape(y, shape=(-1,1,n_class)))
            v_masked = tf.reduce_sum(v_masked, axis=2, name='v_mask') # (B x 16)

            v_reco = layer3( layer2( layer1( v_masked ) ) )

            reco_loss = tf.reduce_mean(tf.square(v_reco - tf.reshape(x, shape=(-1, 784))), name='reco_loss')

        v_len = tf.sqrt(tf.reduce_sum(tf.square(v), axis=1) + eps)

        with tf.name_scope('classif_loss'):
            # loss as proposed in the paper
            l_klass = y * (tf.maximum(zeros((1,1),dtype=float32), MPlus-v_len)**2) + \
                lam * (1-y) * (tf.maximum(zeros((1,1),dtype=float32), v_len-MMinus)**2)

            class_loss = tf.reduce_mean(l_klass, name='loss')

        with tf.name_scope('full_loss'):
            loss = class_loss + reco_loss * reco_loss_importance
            loss_summary = tf.summary.scalar('loss', loss)

    with tf.name_scope('testing'):
        correct_prediction = tf.equal(tf.argmax(v_len, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('optim'):
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("CAPSLog")
        writer.add_graph(graph=sess.graph)

        gstep = 1

        for E in range(epochs):
            for I in range(int(55000 / bsize)):
                X, Y = mnist.train.next_batch(bsize)
                X = X.reshape((bsize, 28, 28, 1))

                _, cl, rl, l_sum = sess.run([train_step, class_loss, reco_loss, loss_summary],
                    feed_dict={x: X, y: Y})
                if I % 5 == 0:
                    print('loss: {2} = {0} + {1}'.format(cl, rl*reco_loss_importance, cl+rl*reco_loss_importance))
                writer.add_summary(l_sum, gstep)
                gstep += 1


if __name__ == '__main__':
    if os.path.exists("CAPSLog"):
        shutil.rmtree("CAPSLog")
    main()