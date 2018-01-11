import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class MnistDemo:

    def __init__(self):
        self._create_placeholder()
        pass

    def _create_placeholder(self):
        self.x=tf.placeholder(tf.float32,shape=[784,1])
        self.y=tf.placeholder(tf.float32,shape=[10,1])

    def _create_embeddings(self):

        w1=tf.Variable(tf.random_normal([784,30],mean=0,stddev=0.1)).initialized_value()
        h1=tf.sigmoid(tf.matmul(tf.transpose(w1),self.x))


        w2 = tf.Variable(tf.random_normal([30,10], mean=0, stddev=0.1)).initialized_value()

        o =tf.sigmoid(tf.matmul(tf.transpose(w2),h1))
        print o
        return o

    def _create_loss(self):
        loss=tf.sqrt(tf.squared_difference(self.y,self._create_embeddings()))
        return  loss

    def _create_optimizer(self):
        opt=tf.train.GradientDescentOptimizer(0.2).minimize(self._create_loss())
        return opt


if __name__=="__main__":

    obj=MnistDemo()
    data = input_data.read_data_sets("./data/mnist", one_hot=True)
    num_batches=int(data.train.num_examples/50)
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in xrange(num_batches):
            x_train,y_train=data.train.next_batch(1)
            x_train=np.transpose(x_train)
            y_train=np.transpose(y_train)
            sess.run(obj._create_optimizer(),feed_dict={obj.x:x_train,obj.y:y_train})
