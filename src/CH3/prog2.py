import tensorflow as tf
import numpy as np


# main dunction to call other functions
def run():
    # cret_graph()
    # mng_graphs()
    # seq_gen()
    # mat_mul()
    # var_place()
    tf_placeholders()
    pass


def cret_graph():
    # as we can crete our own graph int the tesorflow and assign it
    g = tf.Graph()  # creting new empty graph
    print(g)
    print("The TF Default graph : {}", tf.get_default_graph())  # this method will give the default graph.

    # we can also check the associated graph for a perticular node

    # example
    node = tf.constant(10)
    print(node.graph is g)  # this will give the associated graph for the perticular node
    print(node.graph)
    print(node.graph is tf.get_default_graph())


# working with 'with' keyword to manage graphs in tf :
def mng_graphs():
    g1 = tf.get_default_graph()
    g2 = tf.Graph()

    print(g1 is tf.get_default_graph())

    with g2.as_default():
        print(g1 is tf.get_default_graph())

    print(g1 is tf.get_default_graph())


## generating squences

# this will generate the equially spaced sequince of n numbers
def seq_gen():
    sess = tf.InteractiveSession()
    c = tf.linspace(0.0, 4.0, 50)
    print("The {}".format(c.eval()))
    sess.close()


# matrix multiplication
def mat_mul():
    a = tf.constant([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print(a.get_shape())  # get_shape method will give the shape of the object

    b = tf.constant([1, 2, 3])
    print(b.get_shape())

    # to get the name of the attribute we can use the .name attribute
    print(b.name)
    # you can change the dimentions of the vector
    b = tf.expand_dims(b, 1)
    print(b.get_shape())

    # multiplying two matricec
    res = tf.matmul(a, b)

    sess = tf.InteractiveSession()
    print(res.eval())
    # getting the transpose of the matrix
    print(tf.transpose(res.eval()))

    ## grouping the objects to manage the objects
    with tf.Graph().as_default():
        c1 = tf.constant(4)
        with tf.name_scope("prefix_name"):
            c2 = tf.constant(10)
            c3 = tf.constant(11)
    print(c1.name)
    print(c2.name)
    print(c3.name)
    sess.close()


## tensor variables, placeholders and optimization
def var_place():
    # ininital = tf.random_normal((1,5), 0, 1)
    # var = tf.Variable(ininital, name="Variable")
    # print("Pre run Variable : {}".format(var))
    #
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     post_var = sess.run(var)
    # print("Post run Variable : {}".format(post_var))

    init_val = tf.random_normal((1, 5), 0, 1)
    var = tf.Variable(10, name='var')
    print("pre run: \n{}".format(var))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        post_var = sess.run(var)
    print("\npost run: \n{}".format(post_var))


def tf_placeholders():
    x_data = np.random.randn(5, 10)
    w_data = np.random.randn(10, 1)
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=(5, 10))
        w = tf.placeholder(tf.float32, shape=(10, 1))
        b = tf.fill((5, 1), -1.)
        xw = tf.matmul(x, w)
        xwb = xw + b
        s = tf.reduce_max(xwb)
        with tf.Session() as sess:
            outs = sess.run(s, feed_dict={x: x_data, w: w_data})
    print("outs = {}".format(outs))


run()
