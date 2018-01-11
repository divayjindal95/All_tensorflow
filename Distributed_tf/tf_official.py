import  tensorflow as tf

def main():

    cluster = tf.train.ClusterSpec({
        'worker': ['localhost:2223',
                   'localhost:2224'],

        'ps': ['localhost:2222']
    })

    #job="ps"
    job="worker"
    index=1
    server=tf.train.Server(server_or_cluster_def=cluster,job_name=job,task_index=index)

    if job=='ps':
        print "ps joined"
        server.join()

    else:
        print "worker %d"%index

        x1=input("write")
        x2 = input()
        y1 = input()


        with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d/cpu:/gpu:'%index,cluster=cluster,)):

            init = tf.global_variables_initializer()
            x=tf.placeholder(dtype=tf.float32,shape=[2])
            y = tf.placeholder(dtype=tf.float32)
            W=tf.Variable(tf.truncated_normal([2,1],dtype=tf.float32),name="W%d"%index)

            loss= y- tf.reduce_mean(tf.multiply(W,x))
            opt=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

            saver=tf.train.Saver()
            merge=tf.summary.merge_all()

            init=tf.global_variables_initializer()


        sv=tf.train.Supervisor(is_chief=(index==0),
                               logdir="./tf_official",
                               summary_op=merge,init_op=init,
                               saver=saver,
                               )

        with sv.managed_session(server.target) as sess:

            sess.run(init)
            print sess.run(W)
            _,l=sess.run([opt,loss],feed_dict={x:[x1,x2],y:[y1]})
            print l
            print sess.run(x)


if __name__=="__main__":
    main()