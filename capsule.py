import tensorflow as tf
from get_data import get_mini_mnist
import os, shutil
from numpy import zeros, float32, linalg

bsize = 10

def main( args=None ):
    X, _ = get_mini_mnist(bsize=bsize,as_image=True)
    X = X.reshape((-1, 28, 28, 1))

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
                dtype=tf.float32, initializer=tf.initializers.truncated_normal)
            b_conv1 = tf.get_variable('b_conv1', shape=(conv1_fea_maps,), dtype=tf.float32,
                initializer=tf.initializers.zeros)

    with tf.name_scope('conv1'):
        conv1_pre = tf.nn.conv2d(x, W_conv1, strides=conv1_strides, padding='VALID') + b_conv1
        conv1_act = tf.nn.relu(conv1_pre, name='conv1_act')

    with tf.variable_scope(params):
        with tf.variable_scope('Conv2Caps'):
            w_var_name = 'Conv2Caps' + '_W'
            b_var_name = 'Conv2Caps' + '_b'

            W_convcaps = tf.get_variable(w_var_name, dtype=tf.float32,
                shape=(*convcaps_kernel_size, conv1_fea_maps, convcaps_capsule_dim*convcaps_fea_maps),
                initializer=tf.initializers.truncated_normal)
            b_convcaps = tf.get_variable(b_var_name, dtype=tf.float32,
                shape=(convcaps_capsule_dim*convcaps_fea_maps,),
                initializer=tf.initializers.zeros)

    with tf.name_scope('PrimCaps'):
        primecaps_act = tf.nn.relu(
                tf.nn.conv2d(conv1_act, W_convcaps, convcaps_stride, 'VALID') + b_convcaps
            )

        primecaps_act = tf.split(primecaps_act, axis=3, num_or_size_splits=convcaps_fea_maps,
            name='primecaps_act')

    with tf.name_scope('CapsFlatten'):
        primecaps_cap_r = [
            tf.reshape(act, shape=(-1, 6*6, convcaps_capsule_dim))
            for act in primecaps_act
        ]

        primecaps_cap_f = tf.concat(primecaps_cap_r, axis=1)
        primecaps_cap_f = tf.reshape(primecaps_cap_f, 
            shape=(-1, 6*6*convcaps_fea_maps, 1, convcaps_capsule_dim))

    with tf.variable_scope(params):
        with tf.variable_scope('Caps2Caps'):
            W = tf.get_variable('W', dtype=tf.float32, initializer=tf.initializers.random_normal,
                shape=(1, 6*6*convcaps_fea_maps, convcaps_capsule_dim, out_capsule_dim, n_class))

            # bij: the logits for c_ij
            bij = tf.get_variable('bij', dtype=tf.float32, initializer=tf.initializers.zeros,
                shape=(6*6*convcaps_fea_maps, n_class))

    with tf.name_scope('Caps2Digs'):
        # coupling coeff c_ij
        cij = tf.nn.softmax(bij, dim=1)

        uj = []
        with tf.name_scope('pred_vec'):
            for c in range(n_class):
                uj.append( tf.matmul(primecaps_cap_f, tf.tile(W[...,c], [bsize,1,1,1]),
                    name='u_'+str(c)) )

            # full 'prediction vector'; need it in 'routing'
            u = tf.squeeze(tf.stack(uj, axis=4), axis=None, name='u')

        with tf.name_scope('cap_pre_act'):
            s = tf.squeeze(tf.reduce_sum(
                    u * tf.reshape(cij, shape=(1,6*6*convcaps_fea_maps,1,n_class)), axis=1),
                axis=None, name='s')

        with tf.name_scope('cap_act'):
            squash_factor = tf.norm(s, axis=1, keep_dims=True) / (1 + tf.norm(s, axis=1, keep_dims=True) ** 2)
            v = s * squash_factor # auto broad-casting

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("CAPSLog")
        writer.add_graph(graph=sess.graph)

        q = sess.run(v, feed_dict={x: X})
        print(q.shape)
        print(linalg.norm(q, axis=1))

if __name__ == '__main__':
    if os.path.exists("CAPSLog"):
        shutil.rmtree("CAPSLog")
    main()