import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

batch_size=100
x=tf.placeholder(tf.float64,shape=[784,batch_size])
y=tf.placeholder(tf.float64,shape=[10,batch_size])

with tf.name_scope("l2") as scope:
    w1=tf.Variable(tf.random_normal([784,30],mean=0,stddev=0.1,dtype=tf.float64))
    h1=tf.nn.softmax(tf.matmul(tf.transpose(w1),x))

with tf.name_scope("l1") as scope:
    w2 = tf.Variable(tf.random_normal([30,10], mean=0, stddev=0.1,dtype=tf.float64))
    o =tf.nn.softmax(tf.matmul(tf.transpose(w2),h1))

loss=tf.reduce_mean(tf.squared_difference(y,o))
#loss=-tf.reduce_mean(y*tf.log(o)+(1-y)*tf.log(1-o))
tf.summary.scalar("loss",loss)
#loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=o)
tf.summary.histogram("histogram",loss)
opt=tf.train.RMSPropOptimizer(learning_rate=0.3).minimize(loss)


data = input_data.read_data_sets("./data/mnist", one_hot=True)

merged=tf.summary.merge_all()
init=tf.global_variables_initializer()
with tf.Session() as sess:
    mysummary=tf.summary.FileWriter("./assignment1",sess.graph)
    sess.run(init)
    minloss=[]
    num_batches = int(data.train.num_examples / batch_size)
    for i in xrange(num_batches):
        #print i
        x_train,y_train=data.train.next_batch(batch_size)
        x_train=np.transpose(x_train)
        y_train=np.transpose(y_train)
        summ ,_,lsss=sess.run([merged,opt,loss],feed_dict={x:x_train,y:y_train})
        print lsss
        #minloss.append(sss)
        mysummary.add_summary(summ,i)



    num_batches=int(data.test.num_examples/batch_size)
    for i in xrange(num_batches):
        x_test,y_test=data.test.next_batch(batch_size)
        x_test = np.transpose(x_test)
        y_test = np.transpose(y_test)

    mysummary.close()