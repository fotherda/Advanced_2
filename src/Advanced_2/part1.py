from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import Advanced_1.confusionMatrix as cm

from tensorflow.examples.tutorials.mnist import input_data
from Advanced_1.convergenceTester import ConvergenceTester
from Advanced_1.learningRateScheduler import LearningRateScheduler
from Advanced_1.dataBatcher import DataBatcher
from sklearn.metrics import confusion_matrix
from tensorflow.python.ops import rnn
from scipy.misc import toimage



root_dir = 'C:/Users/Dave/Documents/GI13-Advanced/Assignment2';
summaries_dir = root_dir + '/Summaries';
save_dir = root_dir;


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
def print_confusion_matrix_command_line(x, y_, X, Y_, argm_y, argm_y_, sess, name, keep_prob):
    pred_label = sess.run([argm_y, argm_y_], feed_dict={x: X, y_: Y_, keep_prob: 1.0})   
    
    cm = tf.contrib.metrics.confusion_matrix(pred_label[0], pred_label[1], 
                num_classes=None, dtype=tf.int32, name=None, weights=None)   
    
    print( name, ' confusion matrix')
    cm_str = np.array2string(cm.eval(), separator=', ')
    print(str(cm_str).replace('[','').replace(']',''))
    
def print_confusion_matrix(x, y_, X, Y_, argm_y, argm_y_, sess, keep_prob, model_type):

    pred_label = sess.run([argm_y, argm_y_], feed_dict={x: X, y_: Y_, keep_prob: 1.0})   
    
    y_pred = pred_label[0]
    y_true = pred_label[1]
    
    cnf_matrix = confusion_matrix(y_true, y_pred)
    cm_filename = model_type.replace(' ','_') + '.p'
    pickle.dump( cnf_matrix, open( cm_filename, "wb" ) )    
    
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Plot non-normalized confusion matrix
    plt.figure()
    cm.plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title=('Confusion matrix - ' + model_type))    
    plt.show()

def build_network_task1(x, nrecurrent_units, cell, y_):
    
#     W_1 = weight_variable([1, nrecurrent_units])
#     b_1 = bias_variable([nrecurrent_units]) 
#     y = tf.reshape(x, [-1, 1])
#     y = tf.matmul(y, W_1) + b_1
#     y = tf.reshape(y, [-1, x.get_shape()[1].value, nrecurrent_units])
    
#     y_2 = tf.expand_dims(y_1, -1)
#     rnn_outputs, state = rnn.rnn(cell, )  
    raw_rnn_outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
#     val = tf.transpose(raw_rnn_outputs, [1, 0, 2]) # swap batch_size and max_time
#     last_rnn_output = tf.gather(val, int(val.get_shape()[0]) - 1) #just use last value
    last_rnn_output = tf.slice(raw_rnn_outputs, [0, raw_rnn_outputs.get_shape()[1].value - 1, 0], 
                               [-1, 1, raw_rnn_outputs.get_shape()[2].value] )
    last_rnn_output = tf.squeeze(last_rnn_output, 1, name='sliced_rnn_outputs')

    W_2 = weight_variable([nrecurrent_units, nrecurrent_units])
    b_2 = bias_variable([nrecurrent_units])   
    h_2 = tf.nn.relu(tf.matmul(last_rnn_output, W_2) + b_2)    
    
    W_3 = weight_variable([nrecurrent_units, 10])
    b_3 = bias_variable([10])
    y = tf.matmul(h_2, W_3) + b_3
    
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    
    return y, cross_entropy

def build_network_task2(x, nrecurrent_units, cell, y_):
    
    raw_rnn_outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    
    #cut off the last output as it isn't used for prediction
    raw_rnn_outputs = tf.slice(raw_rnn_outputs, [0,0,0], [-1,x.get_shape()[1].value-1, nrecurrent_units], name='pixel_targets')

    W_1 = weight_variable([nrecurrent_units, 2])
    b_1 = bias_variable([2])   
        
    y = tf.reshape(raw_rnn_outputs, [-1, nrecurrent_units])
    y = tf.matmul(y, W_1) + b_1
    y = tf.reshape(y, [-1, raw_rnn_outputs.get_shape()[1].value, 2])

    #don't scale the logits that will done in the loss function
    
    #build target labels as the next pixel
    targets = tf.slice(x, [0,1,0],[-1, x.get_shape()[1].value-1, 1])
    targets = tf.squeeze(targets, 2, name='pixel_targets')
    
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=y))

    return y, cross_entropy
           
def binarize(images, threshold=0.1):
    return (threshold < images).astype('float32')

def add_dimension(images):
    return np.expand_dims(images, -1)

def save_model(session, model_name):
    if not os.path.exists(root_dir + '/model/'):
        os.mkdir(root_dir + '/model/')
    saver = tf.train.Saver(write_version=2)
    save_path = saver.save(session, root_dir + '/model/' + model_name +'.ckpt')
#     print("Model saved in file: %s" % save_path)


def import_data(FLAGS):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    X_train_bin = binarize(mnist.train.images)
    X_test_bin = binarize(mnist.test.images)
    X_train = add_dimension( X_train_bin )
    X_test = add_dimension( X_test_bin )
    
    if FLAGS.model[:2]=='P1':
        y_train = mnist.train.labels
        y_test = mnist.test.labels   
    elif FLAGS.model[:2]=='P2':
        y_train = X_train_bin
        y_test = X_test_bin   

    ns = 0
    if ns > 0:
        X_train = X_train[:ns]
        y_train = y_train[:ns]    
        X_test = X_test[:ns]
        y_test = y_test[:ns]

    npixels = X_train_bin.shape[1]
    X_train_bin = X_train_bin.astype(int)
    X_test_bin = X_test_bin.astype(int)
    X_train_bin = np.delete(X_train_bin, npixels-1, 1)
    X_test_bin = np.delete(X_test_bin, npixels-1, 1)
       
        
    embedding = np.array([0,1,2,3,4,5,6,7,8,9])
    y_train_dense = np.dot(y_train, embedding)  
    y_test_dense = np.dot(y_test, embedding)  

#     rs = np.reshape(mnist.train.images[0], (28,28))
#     toimage(rs).show()
#     rs = np.reshape(X_train[1], (28,28))
#     toimage(rs).show()

    return X_train, y_train_dense.astype(int), X_test, y_test_dense.astype(int)

def run_part1_models(FLAGS):
    print('Tensorflow version: ', tf.VERSION)
    print(sys.path)

    # Hyperparameters
    max_num_epochs = 100
    dropout_val = 0.55
    learning_rate_val = 0.01
    decay = learning_rate_val / 20
    use_peepholes = False; peep_str='' #only for LSTM
    BATCH_SIZE = 128
    
    # Import data
    X_train, y_train, X_test, y_test = import_data(FLAGS)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    x = tf.placeholder(tf.float32, [None, 784, 1], name='x')
    if FLAGS.model[:2]=='P1':
        y_ = tf.placeholder(tf.int32, [None], name='y_')
    elif FLAGS.model[:2]=='P2':
        y_ = tf.placeholder(tf.int32, [None, 783], name='y_')
#     next_pixel = tf.placeholder(tf.int32, [None, 783], name='next_pixel')
    

    if FLAGS.model=='P1_1x32_L':
        num_units = 32
        cell = tf.nn.rnn_cell.LSTMCell(num_units, use_peepholes=use_peepholes)
    elif FLAGS.model=='P1_1x64_L':
        num_units = 64
        cell = tf.nn.rnn_cell.LSTMCell(num_units, use_peepholes=use_peepholes)
    elif FLAGS.model=='P1_1x128_L':
        num_units = 128
        cell = tf.nn.rnn_cell.LSTMCell(num_units, use_peepholes=use_peepholes)
    elif FLAGS.model=='P1_3x32_L':
        num_units = 32
        base_cell = tf.nn.rnn_cell.LSTMCell(num_units, use_peepholes=use_peepholes)
        cell = tf.nn.rnn_cell.MultiRNNCell([base_cell] * 3)
    elif FLAGS.model=='P1_1x32_G':
        num_units = 32
        cell = tf.nn.rnn_cell.GRUCell(num_units)
    elif FLAGS.model=='P1_1x64_G':
        num_units = 64
        cell = tf.nn.rnn_cell.GRUCell(num_units)
    elif FLAGS.model=='P1_1x128_G':
        num_units = 128
        cell = tf.nn.rnn_cell.GRUCell(num_units)
    elif FLAGS.model=='P1_3x32_G':
        num_units = 32
        base_cell = tf.nn.rnn_cell.GRUCell(num_units)
        cell = tf.nn.rnn_cell.MultiRNNCell([base_cell] * 3)

    if FLAGS.model[:2]=='P1':
        y, cross_entropy = build_network_task1(x, num_units, cell, y_ ) 
    elif FLAGS.model[:2]=='P2':
        y, cross_entropy = build_network_task2(x, num_units, cell, y_ ) 
    
    tf.summary.scalar('CrossEntropy', cross_entropy)
        
#     train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # Test trained model
    argm_y = tf.to_int32( tf.argmax(y, 1) )
#     argm_y_ = tf.argmax(y_, 1)
    argm_y_ = y_
    correct_prediction = tf.equal(argm_y, argm_y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    path_arr = [FLAGS.model, "drop{:.1f}".format(dropout_val), peep_str, 'bs' + str(BATCH_SIZE)]

    db = DataBatcher(X_train, y_train)

    with tf.Session() as sess:       
                       
        if FLAGS.use_saved: #Restore saved model       
            model_file_name = FLAGS.saved_model_dir + '/' + FLAGS.model + '.ckpt';    
            saver2restore = tf.train.Saver(write_version=1)
            tf.global_variables_initializer().run()    
            saver2restore.restore(sess, model_file_name)
            
        else: #Train new model
            # Merge all the summaries and write them out to file
            merged = tf.summary.merge_all()
            
            summary_file_name = '/'.join(path_arr)
            dir_name = summaries_dir + '/' + summary_file_name;
            train_writer = tf.summary.FileWriter(dir_name + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(dir_name + '/test')
            
            # init operation
            tf.global_variables_initializer().run()    
            
            conv_tester = ConvergenceTester(0.0005, lookback_window=3) #stop if converged to within 0.05%
            lrs = LearningRateScheduler(decay)
            ntrain = X_train.shape[0]

            # Train
            for epoch in range(max_num_epochs):
                
                for i in range(ntrain // BATCH_SIZE):
#                 for i in range(3):
                    learning_rate_val = lrs.get_learning_rate(epoch, learning_rate_val)
    #                 batch_xs, batch_ys = mnist.train.next_batch(50)
                    batch_xs, batch_ys = db.next_batch(BATCH_SIZE)
    #                 batch_xs, batch_ys = tf.train.shuffle_batch([X_train, y_train], 
    #                             batch_size=BATCH_SIZE, capacity=capacity,
    #                             min_after_dequeue=min_after_dequeue, enqueue_many=True)
    
                    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, 
                                                    learning_rate: learning_rate_val, keep_prob: dropout_val})
             
                if epoch % 1 == 0: #calc intermediate results
                    train_accuracy, train_loss, train_summary = sess.run([accuracy, cross_entropy, merged], feed_dict={x: X_train[:1000], y_: y_train[:1000], keep_prob: 1.0})                                      
                    test_accuracy, test_loss, test_summary = sess.run([accuracy, cross_entropy, merged], feed_dict={x: X_test, y_: y_test, keep_prob: 1.0})
                    print("epoch %d, tr:te accuracy %g : %g loss %g : %g learn rate %f" % (epoch, train_accuracy, test_accuracy, train_loss, test_loss, learning_rate_val))
                                   
                    train_writer.add_summary(train_summary, i)
                    test_writer.add_summary(test_summary, i)
                    
                    if conv_tester.has_converged(test_accuracy):
                        print('converged after ', epoch, ' epochs')
                        break
                        
                #save trained model
                model_file_name = '_'.join(path_arr)+'_'+ str(epoch) #write every epoch
                save_model(sess, model_file_name)
        
            
        #print final results        
        print("Training Error rate:", 1-accuracy.eval({x: X_train, y_: y_train, 
                                                              keep_prob: 1.0}))
        print("Test Error rate:", 1-accuracy.eval({x: X_test, y_: y_test, 
                                                              keep_prob: 1.0}))
        print_confusion_matrix(x, y_, X_train, y_train, argm_y, argm_y_, 
                               sess, keep_prob, FLAGS.model + ' training data')
        print_confusion_matrix(x, y_, X_test, y_test, argm_y, argm_y_, 
                               sess, keep_prob, FLAGS.model + ' test data')

