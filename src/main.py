import tensorflow as tf

## this simply used to print the version of the tensorflow.
print(tf.__version__)


## hello world example in tensorflow

h = tf.constant("hello")
w = tf.constant("world!")

hw = h+w

with tf.Session() as tfss:
    ans = tfss.run(hw)

print(ans)  