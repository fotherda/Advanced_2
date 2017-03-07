from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import Advanced_1.confusionMatrix as cm
import pickle

from tensorflow.examples.tutorials.mnist import input_data
from Advanced_1.dataBatcher import DataBatcher
from Advanced_1.learningRateScheduler import LearningRateScheduler
from Advanced_1.network import Network
from Advanced_1.layer import LinearLayer
from Advanced_1.layer import ReLULayer
from Advanced_1.layer import ConvLayer
from Advanced_1.layer import ConvLayerColumns
from Advanced_1.layer import MaxPoolLayer
from Advanced_1.layer import FlattenLayer
from Advanced_1.layer import CrossEntropyLogits

from Advanced_1.convergenceTester import ConvergenceTester

from scipy.misc import toimage
from sklearn.metrics import confusion_matrix


root_dir = 'C:/Users/Dave/Documents/GI13-Advanced/Assignment1';
summaries_dir = root_dir + '/Summaries';
save_dir = root_dir + '/SavedVariables';

def print_confusion_matrix(network, X, y, model_type):    
    y_pred = network.get_predictions(X)
    y_true = y.argmax(axis=1)
    
    cnf_matrix = confusion_matrix(y_true, y_pred)
    cm_filename = model_type.replace(' ','_') + '.p'
    pickle.dump( cnf_matrix, open( cm_filename, "wb" ) )    

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    cm.plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title=('Confusion matrix - ' + model_type))    
    plt.show()

def part_a(network):
    network.addLayer( LinearLayer(10,(10,784),10) )
    return 'P2_a'
   
def part_b(network):
    network.addLayer( ReLULayer(128,(128,784),128) )
    network.addLayer( LinearLayer(10,(10,128),10) )
    return 'P2_b'
   
def part_c(network):
    network.addLayer( ReLULayer(256,(256,784),256) )
    network.addLayer( ReLULayer(256,(256,256),256) )
    network.addLayer( LinearLayer(10,(10,256),10) )
    return 'P2_c'
   
def part_d(network, X_train, X_test, use_saved=True):
    
    cl1 = ConvLayerColumns(depth=16, filter_size=3, width=28, height=28)
    
    if use_saved:
        X_train_cols = np.load('X_train_cols.npy')
        X_test_cols = np.load('X_test_cols.npy')
    else:
        X_train_pad = cl1.pre_pad_all_images(X_train)
        X_train_cols = cl1.im2col(X_train_pad)
        np.save('X_train_cols', X_train_cols)
        
        X_test_pad = cl1.pre_pad_all_images(X_test)
        X_test_cols = cl1.im2col(X_test_pad)
        np.save('X_test_cols', X_test_cols)
           
    network.addLayer( cl1 )
    network.addLayer( MaxPoolLayer(2, 14, 14, (4,14,14,16), 16 )) 
    network.addLayer( ConvLayer(depth=16, filter_size=3, width=14, height=14) ) 
    network.addLayer( MaxPoolLayer(2, 7, 7, (4,7,7,16), 16) ) 
    network.addLayer( FlattenLayer((16,7,7)) ) 
    network.addLayer( ReLULayer(256, (256,784), 256) ) 
    network.addLayer( LinearLayer(10,(10,256),10) )

    return 'P2_d', X_train_cols, X_test_cols   

   
def run_part2_models(FLAGS):    
        
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels   

    network = Network()
    network.addLossFunction( CrossEntropyLogits() )
    
    if FLAGS.model=='P2_a':
        model_type = part_a(network) 
        data_batcher = mnist.train
    elif FLAGS.model=='P2_b':
        model_type = part_b(network) 
        data_batcher = mnist.train
    elif FLAGS.model=='P2_c':
        model_type = part_c(network) 
        data_batcher = mnist.train
    elif FLAGS.model=='P2_d':
        print('P2:d not implemented')
        exit
        model_type, X_train_cols, X_test_cols = part_d(network, X_train, X_test, use_saved=True)
        data_batcher = DataBatcher(X_train_cols, y_train)
        
#     rs = np.reshape(mnist.test.images[0], (28,28))
#     toimage(rs).show()

    train_accuracy_hist = []
    test_accuracy_hist = []
    conv_tester = ConvergenceTester(0.0005, lookback_window=25) #stop if converged to within 0.01%
    learning_rate = 0.005
    decay = learning_rate / 50000
#     learning_rate = 0.1
#     decay = learning_rate / 2e6
    
    model_filename = FLAGS.saved_model_dir + '/' + "trained_model_" + model_type + ".p"
    
    if FLAGS.use_saved:
        network = pickle.load( open( model_filename, "rb" ) )
        train_accuracy = network.get_accuracy(X_train, y_train)
        test_accuracy = network.get_accuracy(X_test, y_test)
    else:
        lrs = LearningRateScheduler(decay)
        for i in range(50000):
            learning_rate = lrs.get_learning_rate(i, learning_rate)
            batch_xs, batch_ys = data_batcher.next_batch(50)
            train_error = network.run_one_train_epoch(batch_xs, batch_ys, learning_rate)
    
            if i % 100 == 0:
                train_accuracy = network.get_accuracy(X_train, y_train)
                test_accuracy = network.append([i,train_accuracy])
                test_accuracy_hist.append([i,test_accuracy])
    
                print('{0:d} accuracy train test: {1:0.5f} : {2:0.5f} learning rate {3:0.8f}'.
                      format( i, train_accuracy, test_accuracy, learning_rate))
                
                if conv_tester.has_converged(test_accuracy):
                    print('converged after ', i, ' epochs')
                    break
            
        np.savetxt(model_type + "_train_accuracy_hist.csv", train_accuracy_hist, delimiter=",", fmt='%f')
        np.savetxt(model_type + "_test_accuracy_hist.csv", test_accuracy_hist, delimiter=",", fmt='%f')
    
        pickle.dump( network, open( model_filename, "wb" ) )    

    print('Final accuracy train, test: {0:0.5f}, {1:0.5f}'.format( train_accuracy, test_accuracy))
    print_confusion_matrix(network, X_train, y_train, model_type + ' train data')
    print_confusion_matrix(network, X_test, y_test, model_type + ' test data')
    
