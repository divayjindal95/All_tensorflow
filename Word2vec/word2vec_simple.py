import tensorflow as tf
import numpy as np
from process_data import process_data
vocab_size=500
window_size = 1
batch_size = 128
feature_num = 50
num_sampled = 50


def word2vec(batch_data):

    with tf.name_scope("data") as scope:
        x_input = tf.placeholder(tf.int32, shape=[batch_size])
        y_input = tf.placeholder(tf.float64, shape=[batch_size, 1])

    with tf.name_scope("embed_matrix") as scope:
        embed_matrix = tf.Variable(tf.random_uniform([vocab_size, feature_num]
                                                 , -1.0, 1.0))

    with tf.name_scope("loss") as scope:
        embed = tf.nn.embedding_lookup(embed_matrix, x_input)
        nce_weight = tf.Variable(tf.truncated_normal([vocab_size, feature_num], stddev=1.0 / 10))
        nce_bias = tf.Variable(tf.zeros([vocab_size]))
        loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=y_input, inputs=embed, num_sampled=num_sampled,
                       num_classes=vocab_size))
        tf.summary.histogram("loss",loss)
        tf.summary.scalar("loss",loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    merge=tf.summary.merge_all()
    with tf.Session() as sess:
        mysummary=tf.summary.FileWriter("./word2vec_simple",sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in xrange(10):
            center, target = next(batch_data);
            myloss,lsss,_=sess.run([merge,loss, optimizer], feed_dict={x_input: center, y_input: target})
            mysummary.add_summary(myloss)
            print myloss
        mysummary.close()

if __name__ == "__main__":

    batch_data = process_data(vocab_size, batch_size, window_size)
    center, target = next(batch_data);

    word2vec(batch_data)