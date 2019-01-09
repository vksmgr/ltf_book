# This program is about creating the first Tensor graph

import tensorflow as tf

# we will create some  nodes this nodes just provide the values
a = tf.constant(10)
b = tf.constant(20)
c = tf.constant(30)

## now we create some othe nodes with some basic operations
d = tf.multiply(a, b)
e = tf.add(d, c)
f = tf.divide(d, e)

# now we need the create the session to runs the programs
session = tf.Session()
out = session.run(f)
session.close()

# printing the values of the graph
print(out)

## Question 1:

a = tf.constant(10)
b = tf.constant(20)

# First operations
d = a + b

# second
c = a * b

# third
f = d + c

# fourth
e = d * c

# Final
g = f / e

session = tf.Session()
out = session.run(g)
session.close()
print(out)

# Question 2:
a = tf.constant(40)
b = tf.constant(20)

# Op1:
c = tf.multiply(a, b)
print(type(c))

# Op2
d = tf.sin(tf.cast(c, tf.float32))

# Op3
e = d / tf.cast(b, tf.float32)

# creting the session
session = tf.Session()
out = session.run(e)
print(out)
session.close()


