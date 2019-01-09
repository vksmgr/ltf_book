import tensorflow as tf


# main dunction to call other functions
def run():
    # cret_graph()
    mng_graphs()
    pass

def cret_graph():
    # as we can crete our own graph int the tesorflow and assign it
    g = tf.Graph()  # creting new empty graph
    print(g)
    print("The TF Default graph : {}", tf.get_default_graph())      # this method will give the default graph.

    #we can also check the associated graph for a perticular node

    # example
    node = tf.constant(10)
    print(node.graph is g)     # this will give the associated graph for the perticular node
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


run()