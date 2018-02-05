import tensorflow as tf
assert True == False
embeddings = tf.Variable(
    tf.random_uniform([10, 10], -1.0, 1.0))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(sess.run(embeddings))
