'''
Created on 28 Jan 2017

@author: Dave
'''

import matplotlib.pyplot as plt
import numpy as np
import itertools
import pickle

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compare_confusion_matrices():
    labels = ['a','b','c','d']
    cm_list = []
    
    for label in labels:
        model_filename = 'P1_'+ label + '_training_data.p'
        cm = pickle.load( open( model_filename, "rb" ) )
        np.fill_diagonal(cm, 0)
        cm = cm/cm.sum(axis=1)[:,None] #normalize each row
        cm_list.append( cm.flatten() )
        
    for label in labels[:3]:
        model_filename = 'P2_'+ label + '_train_data.p'
        cm = pickle.load( open( model_filename, "rb" ) )
        np.fill_diagonal(cm, 0)
        cm = cm/cm.sum(axis=1)[:,None] #normalize each row
        cm_list.append( cm.flatten() )
        
    corr_mtx = np.zeros((len(cm_list),len(cm_list)))    
        
    for i in range(len(cm_list)):
        for j in range(len(cm_list)):         
            rho = np.corrcoef(cm_list[i], cm_list[j], rowvar=0)
            corr_mtx[i,j] = rho[0,1]
    
    np.set_printoptions(precision=2)
    corr_mtx_str = np.array2string(corr_mtx, separator=', ')
    print(str(corr_mtx_str).replace('[','').replace(']',''))       
    return
