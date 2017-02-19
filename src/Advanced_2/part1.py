from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pickle
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import matplotlib.pyplot as plt

from timeit import default_timer as timer

from tensorflow.examples.tutorials.mnist import input_data
from Advanced_1.convergenceTester import ConvergenceTester
from Advanced_1.learningRateScheduler import LearningRateScheduler
from Advanced_1.dataBatcher import DataBatcher
from Advanced_2.visualize import *
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
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)
  
def zero_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
  
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

    W_2 = weight_variable([nrecurrent_units, 100])
    b_2 = bias_variable([100])   
    h_2 = tf.nn.relu(tf.matmul(last_rnn_output, W_2) + b_2)    
    
    W_3 = weight_variable([100, 10])
    b_3 = bias_variable([10])
    y = tf.matmul(h_2, W_3) + b_3
    
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    
    return y, cross_entropy

def build_network_task2(x, nrecurrent_units, cell, y_):
    
    raw_rnn_outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32) # batch_size x 783 x 32
    
    W_1 = zero_variable([nrecurrent_units, 1])
    b_1 = zero_variable([1])   
#     W_1 = weight_variable([nrecurrent_units, 1])
#     b_1 = bias_variable([1])   
         
    reshaped_outputs = tf.reshape(raw_rnn_outputs, [-1, nrecurrent_units])
    logits = tf.matmul(reshaped_outputs, W_1) + b_1
    logits = tf.reshape(logits, [-1, raw_rnn_outputs.get_shape()[1].value, 1])
    logits = tf.squeeze(logits, 2, name='logits')
    epsilon = tf.constant(value=0.0000001)
    logits = logits + epsilon
    
    tf.check_numerics(logits,'numerical problem with logits')
#     if(np.isnan(logits.eval()).any()):
#         print('nan in logits')
#         print(logits)
    
    y = tf.nn.sigmoid(logits, name='sigmoid_outputs')
    target_float = tf.to_float(y_)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, target_float))
    
    #don't scale the logits that will done in the loss function
        
#     cross_entropy = tf.reduce_mean(
#         tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits))

    return y, cross_entropy
          
def show_all_variables():
    total_count = 0
    for idx, op in enumerate(tf.trainable_variables()):
        shape = op.get_shape()
        count = np.prod(shape)
        print("[%2d] %s %s = %s" % (idx, op.name, shape, count))
        total_count += int(count)
    print("[Total] variable size: %s" % "{:,}".format(total_count))
 
def binarize(images, threshold=0.1):
    return (threshold < images).astype('float32')

def add_dimension(images):
    return np.expand_dims(images, -1)

def save_model(session, model_name):
    if not os.path.exists(root_dir + '/model/'):
        os.mkdir(root_dir + '/model/')
    saver = tf.train.Saver(write_version=1)
    save_path = saver.save(session, root_dir + '/model/' + model_name +'.ckpt')
#     print("Model saved in file: %s" % save_path)

def import_data(FLAGS, num_train_examples):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    X_train_bin = binarize(mnist.train.images)
    X_test_bin = binarize(mnist.test.images)
    X_train = add_dimension( X_train_bin )
    X_test = add_dimension( X_test_bin )
    
    if FLAGS.model[:2]=='P1':
        y_train = mnist.train.labels
        y_test = mnist.test.labels   
        embedding = np.array([0,1,2,3,4,5,6,7,8,9])
        y_train = (np.dot(y_train, embedding)).astype(int)  
        y_test = (np.dot(y_test, embedding)).astype(int)  
    elif FLAGS.model[:2]=='P2':
        y_train = X_train_bin
        y_test = X_test_bin   
        y_train = np.delete(y_train, 0, 1) #remove first pixel
        y_test = np.delete(y_test, 0, 1)

        npixels = X_train_bin.shape[1]
        X_train = X_train_bin.astype(int)
        X_test = X_test_bin.astype(int)
        if not FLAGS.use_saved:
            X_train = np.delete(X_train, npixels-1, 1) #remove last pixel
            X_test = np.delete(X_test, npixels-1, 1)   
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        
    if num_train_examples > 0:
        X_train = X_train[:num_train_examples]
        y_train = y_train[:num_train_examples]    
#         X_test = X_test[:num_train_examples]
#         y_test = y_test[:num_train_examples]

#     rs = np.reshape(mnist.train.images[0], (28,28))
#     toimage(rs).show()
#     rs = np.reshape(X_train[1], (28,28))
#     toimage(rs).show()

    print('Imported data')
    return X_train, y_train, X_test, y_test

class Parser():

    def __init__(self, model):
        self._model = model
        self._task = model[:2]
        self._num_layers = int(model[3])
        self._num_units = int(model[5:8])
        self._cell_type = model[9]


def run_part1_models(FLAGS):
    print('Tensorflow version: ', tf.VERSION)
    print('PYTHON_PATH: ',sys.path)

    ########## Hyperparameters #################
    max_num_epochs = 70
    dropout_val = 0.5
    learning_rate_val = 0.01
    
    decay = learning_rate_val / 2e8
    use_peepholes = False; peep_str='' #only for LSTM
    BATCH_SIZE = 32
    num_train_examples = 0
    
    image_size = 784
    # Import data
    X_train, y_train, X_test, y_test = import_data(FLAGS, num_train_examples)

    #Build different model types
    ps = Parser(FLAGS.model)
    
    if ps._cell_type=='L':
        cell = tf.nn.rnn_cell.LSTMCell(ps._num_units, use_peepholes=use_peepholes)
    elif ps._cell_type=='G':
        cell = tf.nn.rnn_cell.GRUCell(ps._num_units)

    if ps._num_layers>1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * ps._num_layers)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    if ps._task=='P1':
        x = tf.placeholder(tf.float32, [None, image_size, 1], name='x')
        y_ = tf.placeholder(tf.int32, [None], name='y_')
        y, cross_entropy = build_network_task1(x, ps._num_units, cell, y_ ) 
    elif ps._task=='P2':
        x = tf.placeholder(tf.float32, [None, image_size-1, 1], name='x')
        y_ = tf.placeholder(tf.int32, [None, image_size-1], name='y_')
        y, cross_entropy = build_network_task2(x, ps._num_units, cell, y_ )     

        
#     train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # Test trained model
    if ps._task=='P1':
        argm_y = tf.to_int32( tf.argmax(y, 1) )
        correct_prediction = tf.equal(argm_y, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
#     elif ps._task=='P2':
#         argm_y = tf.to_int32( tf.argmax(y, 2) )
#         argm_y = tf.to_int32( tf.greater(y, 0.5) )

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('CrossEntropy', cross_entropy)
    path_arr = [FLAGS.model, "drop{:.1f}".format(dropout_val), peep_str, 'bs' + str(BATCH_SIZE),
                "lr{:.2g}".format(learning_rate_val), "nt{:d}".format(num_train_examples)]

#     show_all_variables()
    db = DataBatcher(X_train, y_train)

    with tf.Session() as sess:  
#         sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#         sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)     
                       
        if FLAGS.use_saved: #Restore saved model   
#             fn = 'P2_1x032_G_drop0.5__bs32_lr0.005_nt0_38'
            fn = 'P2_1x064_G_drop0.5__bs32_lr0.005_nt0_32'  
            model_file_name = FLAGS.saved_model_dir + '/' + fn + '.ckpt'    
#             model_file_name = FLAGS.saved_model_dir + '/' + FLAGS.model + '.ckpt'    
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
            
            conv_tester = ConvergenceTester(0.0005, lookback_window=5, decreasing=True) #stop if converged to within 0.05%
            lrs = LearningRateScheduler(decay)
            ntrain = X_train.shape[0]

            # Train
            for epoch in range(max_num_epochs):
                start = timer()
           
                for i in range(ntrain // BATCH_SIZE):
                    learning_rate_val = lrs.get_learning_rate(epoch, learning_rate_val)
                    batch_xs, batch_ys = db.next_batch(BATCH_SIZE)
                    _, pred = sess.run([train_step,y], feed_dict={x: batch_xs, y_: batch_ys, 
                                                    learning_rate: learning_rate_val, keep_prob: dropout_val})
                end = timer()
             
                if epoch % 1 == 0: #calc intermediate results
     
                    if ps._task=='P1':
                        train_accuracy, train_loss, train_summary = sess.run([accuracy, cross_entropy, merged], feed_dict={x: X_train[:5000], y_: y_train[:5000], learning_rate: learning_rate_val, keep_prob: 1.0})                                      
                        test_accuracy, test_loss, test_summary = sess.run([accuracy, cross_entropy, merged], feed_dict={x: X_test, y_: y_test, learning_rate: learning_rate_val, keep_prob: 1.0})
                        print("\nepoch %d, tr:te accuracy %g : %g loss %g : %g lr %g" % (epoch, train_accuracy, test_accuracy, train_loss, test_loss, learning_rate_val))
                                   
                    elif ps._task=='P2':
                        train_loss, train_summary = sess.run([cross_entropy, merged], feed_dict={x: X_train[:1000], y_: y_train[:1000], learning_rate: learning_rate_val, keep_prob: 1.0})                                      
                        test_loss, test_summary = sess.run([cross_entropy, merged], feed_dict={x: X_test, y_: y_test, learning_rate: learning_rate_val, keep_prob: 1.0})
                        print("\nepoch %d, loss %g : %g lr %g et %g" % (epoch, train_loss, test_loss, learning_rate_val, end-start))
                                   
                    if np.isnan(train_loss) or np.isnan(test_loss):
                        exit()
                    
                    train_writer.add_summary(train_summary, i)
                    test_writer.add_summary(test_summary, i)
                    
                    if conv_tester.has_converged(test_loss):
                        print('converged after ', epoch, ' epochs')
                        break
                        
                #save trained model
                model_file_name = '_'.join(path_arr)+'_'+ str(epoch) #write every epoch
                save_model(sess, model_file_name)
            exit()
            
        #print final results        
#         print("Training Error rate:", 1-accuracy.eval({x: X_train, y_: y_train, keep_prob: 1.0}))
#         print("Test Error rate:", 1-accuracy.eval({x: X_test, y_: y_test, keep_prob: 1.0}))

#         in_painting(y, x, y_, keep_prob, sess)

        #prediction
        nsamples = 100
        mask_length = 300
        db2 = DataBatcher(X_test, y_test)
        batch_xs, batch_ys = db2.next_batch(nsamples)
        ground_truth_images = np.copy(batch_xs)
                    
        pixel_preds = np.zeros((nsamples, mask_length), dtype='float32')            
        pixel_gt = np.zeros((nsamples, mask_length), dtype='float32')            
        pixel_idx = batch_xs.shape[1] - mask_length - 1                  
        for i in range(mask_length):
            pred = sess.run(y, feed_dict={x: batch_xs[:,:783], y_: batch_ys[:,:783], keep_prob: 1.0})

            pixel_preds[:, i] = pred[:, pixel_idx]
            pixel_idx += 1
            pixel_gt[:, i] = batch_xs[:, pixel_idx, 0]
            batch_xs[:, pixel_idx, 0] = pixel_preds[:, i]
        
        get_cross_entropy(ground_truth_images, pixel_preds, pixel_gt)    
        
        
def get_in_paintings(images, gt_images, nip):
    
#     images_ip = np.copy( np.expand_dims(images, 1) ) 
    images_ip = np.zeros( (images.shape[0], nip, images.shape[1]) ) 
    gt_at_mp = np.zeros(gt_images.shape[0])
    
    for i, img in enumerate(images):
        mps = np.where(img<0)
        
        for s in range(nip):
            images_ip[i,s,:] = np.copy(img)
        
        for mp in mps:
            images_ip[i,0,mp] = 0
            images_ip[i,1,mp] = 1
        
        gt_at_mp[i] = gt_images[i,mps]
        
    return images_ip, gt_at_mp    
        
def calc_log_p(preds, gt_at_mp):
    
    nip = preds.shape[1]
    nimgs = preds.shape[0]
    npixels = preds.shape[2]
    
    for i in range(nimgs):
        log_probs = np.zeros((nip))
        for s in range(nip):
            for p in range(npixels):
                log_probs[s] += np.log(preds[i,s,p])

        max_prob_idx = np.argmax(log_probs)
        gt = gt_at_mp[i]

def get_xent_for_mp(gt_idx, sample_idx):
    
    xent = gt*np.log()
    
                
def in_painting(y, x, y_, keep_prob, sess):   
    dataset = np.load(root_dir + '/one_pixel_inpainting.npy')    
#     dataset = np.load(root_dir + '/2X2_pixels_inpainting.npy')
    
    images    = dataset[0]
    gt_images = dataset[1] 
    
    nip = 2
    
    images_ip, gt_at_mp = get_in_paintings(images, gt_images, nip)
    
    images_ip = np.delete(images_ip, images_ip.shape[2]-1, 2) #remove last pixel
            
    
    images_ip_rs = np.reshape(images_ip, [images_ip.shape[0] * images_ip.shape[1], images_ip.shape[2]])
    images_ip_rs = np.expand_dims(images_ip_rs, axis=2)
        
    preds = sess.run(y, feed_dict={x: images_ip_rs, keep_prob: 1.0})

    preds_rs = tf.reshape(preds, images_ip.shape).eval()
    
    calc_log_p(preds_rs, gt_at_mp)
    
    
    
    
    
#     for SampleID in np.random.randint(nSamples,size=3):
#         plt.figure()
#         plt.subplot(1,2,1)
#         plt.imshow(np.reshape(gt_images[SampleID],(28,28)), interpolation='None',vmin=-1, vmax=1)
#         plt.title("GT image")
#         plt.subplot(1,2,2)
#         plt.imshow(np.reshape(images[SampleID],(28,28)), interpolation='None',vmin=-1, vmax=1)
#         plt.title("One pixel missing image")
#         plt.savefig("sample_"+str(SampleID)+"_one_pixel_inpainting.png")
    
    
    
    