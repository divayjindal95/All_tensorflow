import tensorflow as tf
import numpy as np
import pandas as pd

class ModelName:

    def __init__(self):
        pass

    def _create_placeholder(self):
        raise NotImplementedError

    def _create_variable(self):
        raise NotImplementedError

    def create_loss(self):
        raise NotImplementedError

    def create_optimizer(self):
        raise NotImplementedError

    def create_summaries(self):
        raise NotImplementedError

    def create_graph(self):
        raise NotImplementedError


def train_model(model,batch_data):

    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("directoryname", sess.graph)

        # ckpt = tf.train.get_checkpoint_state(os.path.dirname('./4thjuly/checkpoint'))
        # if ckpt and ckpt.model_checkpoint_path:
        #    saver.restore(sess, ckpt.model_checkpoint_path)
        #config = projector.ProjectorConfig()
        #embedding = config.embeddings.add()
        #embedding.tensor_name = "name of tensor you want to view"
        #embedding.metadata_path = "metadata path ih logdir"
        #projector.visualize_embeddings(writer, config)


        '''
        Implement here what all you want tf to evauluate
        '''

        saver.save(sess,"checkpointname")

def data_processing():
    raise NotImplementedError

if __name__ == "__main__":
    pass


