import tensorflow as tf
import numpy as np
from process_data import process_data
from process_data import make_dir
from tensorflow.contrib.tensorboard.plugins import projector
import os

vocab_size=50000
window_size = 1
batch_size = 128
feature_num = 128
num_sampled = 50
learning_rate=0.1
skip_step=5

class Model:

    def __init__(self):
        pass

    def create_placeholder(self):
        self.x_input = tf.placeholder(tf.int32, shape=[batch_size])
        self.y_input = tf.placeholder(tf.float64, shape=[batch_size, 1])

    def create_embedding(self):
        with tf.name_scope("embed_matrix") as scope:
            self.embed_matrix = tf.Variable(tf.random_uniform([vocab_size, feature_num]
                                                 , -1.0, 1.0))
    def create_loss(self):
        with tf.name_scope("loss") as scope:
            embed = tf.nn.embedding_lookup(self.embed_matrix, self.x_input)
            nce_weight = tf.Variable(tf.truncated_normal([vocab_size, feature_num], stddev=1.0 / 10))
            nce_bias = tf.Variable(tf.zeros([vocab_size]))
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=self.y_input,inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocab_size))


    def create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

    def create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.histogram("loss", self.loss)
            tf.summary.scalar("loss", self.loss)
            self.mergeop=tf.summary.merge_all()

    def create_graph(self):
        self.create_placeholder()
        self.create_embedding()
        self.create_loss()
        self.create_optimizer()
        self.create_summaries()

def train_model(model,batch_data):
    saver=tf.train.Saver()
    make_dir("./checkpoints")

    with tf.Session() as sess:
        mysummary = tf.summary.FileWriter("./word2vec_class", sess.graph)
        sess.run(tf.global_variables_initializer())
        #ckpt=tf.train.get_checkpoint_state(os.path.dirname("./checkpoints/checkpoint"))
        #if ckpt and ckpt.model

        for i in xrange(10):
            center, target = next(batch_data);
            myloss, lsss, _ = sess.run([model.mergeop, model.loss, model.optimizer], feed_dict={model.x_input: center, model.y_input: target})
            mysummary.add_summary(myloss)

            if i%skip_step==0:
                saver.save(sess,"./checkpoints/word2vec_class")

        matrix=sess.run(model.embed_matrix)
        embed_again=tf.Variable(matrix[:10],name="embed_again")
        sess.run(embed_again.initializer)
        config=projector.ProjectorConfig()
        summary_writer=tf.summary.FileWriter("./processed")
        embedding=config.embeddings.add()
        embedding.tensor_name=embed_again.name
        embedding.metadata_path='./processed/vocab_1000.tsv'
        projector.visualize_embeddings(summary_writer,config)
        saver_embed=tf.train.Saver([embed_again])
        saver_embed.save(sess,"./processed/model.ckpt",1)



if __name__ == "__main__":

    batch_data=process_data(vocab_size=vocab_size,batch_size=batch_size,skip_window=window_size)

    model=Model()
    model.create_graph()
    train_model(model,batch_data)