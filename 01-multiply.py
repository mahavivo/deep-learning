import tensorflow as tf

a =tf.placeholder("float")

b =tf.placeholder("float")

y = tf.multiply(a,b)

sess = tf.Session()

file_writer = tf.summary.FileWriter('/logs', sess.graph)

print(sess.run(y, feed_dict={a: 3, b: 3}))