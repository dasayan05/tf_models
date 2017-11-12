import tensorflow as tf

# meta information
TOTAL_SAMPLES = 50
BATCH_SIZE = 20
N_BATCHES = TOTAL_SAMPLES/BATCH_SIZE
SEQ_LEN = 5
INP_DIM, HID_DIM, OUT_DIM = 10, 6, 3

# Generate relevant data
from numpy.random import randn
X = randn(TOTAL_SAMPLES, SEQ_LEN, INP_DIM)
Y = randn(TOTAL_SAMPLES, SEQ_LEN, OUT_DIM)

# placeholders
x_ = tf.placeholder(tf.float64, shape=(None,SEQ_LEN,INP_DIM), name='x')
y_ = tf.placeholder(tf.float64, shape=(None,SEQ_LEN,OUT_DIM), name='y')
Wxh = tf.get_variable('Wxh', shape=(INP_DIM,HID_DIM), dtype=tf.float64, initializer=tf.initializers.random_normal)
Whh = tf.get_variable('Whh', shape=(HID_DIM,HID_DIM), dtype=tf.float64, initializer=tf.initializers.random_normal)
Who = tf.get_variable('Who', shape=(HID_DIM,OUT_DIM), dtype=tf.float64, initializer=tf.initializers.random_normal)

states = [tf.placeholder(tf.float64, shape=(None,HID_DIM), name='h0')]
outputs = []

x = tf.unstack(x_, axis=1)

for l in range(SEQ_LEN):
    states.append(tf.tanh(tf.matmul(x[l],Wxh) + tf.matmul(states[-1],Whh)))
    outputs.append(tf.matmul(states[-1],Who))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("RNNLog")
    writer.add_graph(sess.graph)

    states_outputs = sess.run(states + outputs, feed_dict={x_:X, 'h0:0':randn(TOTAL_SAMPLES,HID_DIM)})
    for one_vec in states_outputs:
        print(one_vec.shape)