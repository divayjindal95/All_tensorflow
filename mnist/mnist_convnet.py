from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

class Model:

    def __init__(self):
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 128
        self.SKIP_STEP = 10
        self.DROPOUT = 0.75
        self.N_EPOCHS = 10
        self.N_CLASSES=10


    def _create_placeholder(self):
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, [None, 784], name="X_placeholder")
            self.Y = tf.placeholder(tf.float32, [None, 10], name="Y_placeholder")
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_network(self):

        with tf.variable_scope('conv1') as scope:
            images = tf.reshape(self.X, shape=[-1, 28, 28, 1])
            kernel = tf.get_variable('kernel', [5, 5, 1, 32],
                                     initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases', [32],
                                     initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
            conv1 = tf.nn.relu(conv + biases, name=scope.name)

        with tf.variable_scope('pool1') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')

        with tf.variable_scope('conv2') as scope:
            kernel = tf.get_variable('kernels', [5, 5, 32, 64],
                                     initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases', [64],
                                     initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
            conv2 = tf.nn.relu(conv + biases, name=scope.name)

        with tf.variable_scope('pool2') as scope:
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')

        with tf.variable_scope('fc') as scope:
            input_features = 7 * 7 * 64
            w = tf.get_variable('weights', [input_features, 1024],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [1024],
                                initializer=tf.constant_initializer(0.0))

            pool2 = tf.reshape(pool2, [-1, input_features])
            fc = tf.nn.relu(tf.matmul(pool2, w) + b, name='relu')


        with tf.variable_scope('softmax_linear') as scope:
            w = tf.get_variable('weights', [1024, self.N_CLASSES],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [self.N_CLASSES],
                                initializer=tf.random_normal_initializer())
            self.logits = tf.matmul(fc, w) + b

    def _create_loss(self):
        entropy=tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,logits=self.logits)
        #entropy=tf.losses.mean_squared_error(labels=self.Y,predictions=tf.nn.softmax(self.logits))
        self.loss=tf.reduce_mean(entropy)

    def _create_optimizer(self):
        self.optimizer=tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss,self.global_step)

    def _create_summaries(self):
        tf.summary.scalar("loss",self.loss)
        self.merge=tf.summary.merge_all()

    def _create_graph(self):
        self._create_placeholder()
        self._create_network()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()


def train_model(model,mnist):

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./mnist_convnet", sess.graph)

        ckpt=tf.train.get_checkpoint_state("./checkpoints/mnist_convnet/checkpoint")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)

        initial_step = model.global_step.eval()

        n_batches = int(mnist.train.num_examples / model.BATCH_SIZE)

        total_loss = 0.0
        for index in range(initial_step,n_batches * model.N_EPOCHS):
            X_batch, Y_batch = mnist.train.next_batch(model.BATCH_SIZE)
            _, loss_batch, summary,logits_batch = sess.run([model.optimizer, model.loss,model.merge,model.logits],feed_dict = {model.X: X_batch, model.Y: Y_batch})
            writer.add_summary(summary, global_step=index)
            total_loss += loss_batch

            preds = tf.nn.softmax(logits_batch)
            #print preds.eval()[0],Y_batch[0]
            #print logits_batch[0]
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))


            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds = sess.run(accuracy)
            #print(total_correct_preds,loss_batch)
            #return
            if (index + 1) % model.SKIP_STEP == 0:
                #print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / model.SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, './checkpoints/mnist_convnet/mnist-convnet', index)
                print(total_correct_preds, loss_batch)

        print("Optimization Finished!")


        # test the model
        n_batches = int(mnist.test.num_examples / model.BATCH_SIZE)
        total_correct_preds = 0
        for i in range(n_batches):
            X_batch, Y_batch = mnist.test.next_batch(model.BATCH_SIZE)
            _, loss_batch,logits_batch = sess.run([model.optimizer,model.loss,model.logits],feed_dict = {model.X: X_batch, model.Y: Y_batch})
            preds = tf.nn.softmax(logits_batch)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += sess.run(accuracy)
            print total_correct_preds
        print("Accuracy {0}".format(total_correct_preds / mnist.test.num_examples))





if __name__ == "__main__":
    mnist = input_data.read_data_sets("./data/mnist", one_hot=True)
    model=Model()
    model._create_graph()
    train_model(model,mnist)
