from datetime import datetime
import os

import tensorflow as tf
import numpy as np
from MiniBatchSet import MiniBatchSet


def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def restore_train_state():
    train_snapshot_path = "/usr/local/projects/python/ml/Data/Temp/train_snapshot.ckpt"
    if not os.path.exists(train_snapshot_path):
        return 1

    with open(train_snapshot_path, 'r') as f:
        return int(f.readline().split('=')[1])

def save_train_state(epoch):
    train_snapshot_path = "/usr/local/projects/python/ml/Data/Temp/train_snapshot.ckpt"

    with open(train_snapshot_path, 'w') as f:
        return f.writelines(["EPOCH=" + str(epoch)])

class TensorFLow(object):
    """description of class"""

    def __init__(self):
        pass

    def activate(self):
        number_of_iterations = 1000000

        sess = tf.InteractiveSession()

        tf_snapshot_path = "/usr/local/projects/python/ml/Data/Temp/tf_snapshot.ckpt"
        bs_snapshot_path = "/usr/local/projects/python/ml/Data/Temp/bs_snapshot.json"

        x = tf.placeholder(tf.float64, shape=[None, 8])
        y_ = tf.placeholder(tf.float64, shape=[None, 1])

        W_layer1 = weight_variable([8, 343])
        b_layer1 = bias_variable([343])
        h_layer1 = tf.nn.relu(tf.matmul(x, W_layer1) + b_layer1)

        W_layer2 = weight_variable([343, 1024])
        b_layer2 = bias_variable([1024])
        h_layer2 = tf.nn.relu(tf.matmul(h_layer1, W_layer2) + b_layer2)

        W_layer3 = weight_variable([1024, 5000])
        b_layer3 = bias_variable([5000])
        h_layer3 = tf.nn.relu(tf.matmul(h_layer2, W_layer3) + b_layer3)

        W_readout = weight_variable([5000, 1])
        b_readout = bias_variable([1])
        y_readout = tf.nn.relu(tf.matmul(h_layer3, W_readout) + b_readout)

        print(">--[INIT] [Variables defined]")  

        cross_entropy = tf.reduce_mean(tf.square(y_ - y_readout))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.less(tf.abs(y_readout - y_), 0.000001)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

        print(">--[INIT] [Model is defined]") 

        #basepath = "/mnt/data1/all_features/unpacked"
        basepath = "/usr/local/projects/python/ml/Data/Transformed/"
        
        #Batchlist checkpoint is restored individually since batch selection is in fully random mode
        if os.path.exists(bs_snapshot_path):
            print(">--[INIT] [Loading the mini-batch set from existing snapshot.]")
            batch_list = MiniBatchSet.from_snapshot(bs_snapshot_path)
        else:
            print(">--[INIT] [Started building mini-batch generator.]")
            batch_list = MiniBatchSet(file_paths_or_folder=basepath, batch_size=1000, random_distribution=True)
                    
        print(">--[INIT] [Mini-batch generator is setup.]")

        saver = tf.train.Saver()

        #Try finding any previous saved sessions
        if os.path.exists(tf_snapshot_path):
            saver.restore(sess, tf_snapshot_path)
            first_epoch = restore_train_state()
            print(">--[INIT] [Model was restored from previously saved snapshot]") 
        else:
            first_epoch = 1
            sess.run(tf.initialize_all_variables())
            print(">--[INIT] [Variables fully initialized]") 
                    
        batch_list.save_snapshot(bs_snapshot_path)
        print(">--[INIT] [Savers are setup.]")

        print(">--[INIT] [Started learning from epoch %d]"%first_epoch)

        for epoch in range(first_epoch, number_of_iterations + 1):
            epoch_start = datetime.now()

            batch = np.matrix(next(batch_list))
            batch_x = np.asarray(batch[:,0:8])
            batch_y = np.asarray(batch[:,8])

            if (epoch)%10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y})
                cost = cross_entropy.eval(feed_dict={x:batch_x, y_: batch_y})
                print(">--[EPOCH-%d] [Accuracy = %g] [Cost = %g]"%(epoch, train_accuracy, cost))
                #print("h(theta): ")
                #for i in range(10):
                #    print(">>-->>{0}".format(diffs.eval(feed_dict={x:batch_x, y_: batch_y})))
                #break

            train_step.run(feed_dict={x: batch_x, y_: batch_y})

            if (epoch)%10 == 0:
                minutes_passed = (datetime.now() - epoch_start).total_seconds()/60
                eta = round(((number_of_iterations - epoch) * minutes_passed)/epoch, 2)
                eta_phrase = str(eta) + " minute(s)"

                if eta > 1440:
                    eta_phrase = str(round(eta / 1440, 2)) + " day(s)"
                elif eta > 60:
                    eta_phrase = str(round(eta / 60, 2)) + " hour(s)"

                print(">--[EPOCH-%d] [ETA = %s]"%(epoch, eta_phrase))
                saver.save(sess, tf_snapshot_path)
                save_train_state(epoch)
                print(">--[EPOCH-%d] [Saved]"%(epoch))
             

if __name__ == "__main__":
    trainer = TensorFLow()
    trainer.activate()