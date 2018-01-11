import  tensorflow as tf

cluster1=tf.train.ClusterSpec({
    'worker':['localhost:2222',
              'localhost:2223'],

    'ps':['localhost:2221']
})


c=tf.constant(12)

with tf.device('/job:ps/task:0'):
    x=tf.Variable(tf.random_normal([3,1]))

with tf.device("/job:worker/task:0"):
    y=x*2


with tf.device("/job:worker/task:1"):
    y=x*2


job='worker'
server=tf.train.Server(cluster1,job_name=job,task_index=1)

if job=='ps':
    print "ps joined"
    server.join()

else :
    print "worker working"
    with tf.Session(server.target) as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        print sess.run(x)
        print sess.run(y)

