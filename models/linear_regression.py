import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def runLR():
    x_data = np.random.randn(2000, 3)  # randomly generating the data with 3 features

    w_real = [0.3, 0.5, 0.1]  # weight vector

    b_real = -0.2  # bias

    noise = np.random.randn(1, 2000) * 0.1

    # our linear regression model
    y_data = np.matmul(w_real, x_data.T) + b_real + noise

    # estimating our weights so that the input lable map asclose as the output

    NUM_STEPS = 100

    g = tf.Graph()
    wb_ = []
    with g.as_default():
        x = tf.placeholder(tf.float32, shape=[None, 3])
        y_true = tf.placeholder(tf.float32, shape=None)

        with tf.name_scope('inference') as scope:
            w = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weights')
            b = tf.Variable(0, dtype=tf.float32, name='bias')

            y_pred = tf.matmul(w, tf.transpose(x)) + b

        with tf.name_scope('loss') as scope:
            loss = tf.reduce_mean(tf.square(y_true - y_pred))

        with tf.name_scope('train') as scope:
            learning_rate = 0.5
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for step in range(NUM_STEPS):
                sess.run(train, {x: x_data, y_true: y_data})

                if step % 5 == 0:
                    print(step, sess.run([w, b]))
                    wb_.append(sess.run([w, b]))
            print(10, sess.run([w, b]))
