'''
Created on 25 Jan 2017

@author: Dave
'''

import numpy as np
import abc

class Layer():
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_units, weights, biases):
        self.n_units = n_units
        self.x = None
        self.y = np.zeros(n_units) #outputs / activations
        self.dL_dW = None
        self.dL_db = None
        if not weights is None:
            self.W = (np.random.rand(*weights) - 0.5)/10
        else:
            self.W = None
            
        if biases:
            self.b = np.zeros(n_units)
        else:
            self.b = None
        self.zero_gradients()

        
    @abc.abstractmethod
    def foward_pass(self, x):
        return

    @abc.abstractmethod
    def backward_pass(self, dL_dy):
        return
        
    @abc.abstractmethod
    def param_gradients(self, dL_dy):
        return
        
    def zero_gradients(self, ):
        if not self.W is None:
            self.dL_dW = np.zeros( self.W.shape );
        if not self.b is None:
            self.dL_db = np.zeros( self.n_units );

    def update_parameters(self, learning_rate):
        self.W -= learning_rate * self.dL_dW;
        self.b -= learning_rate * self.dL_db;
     
        
class LinearLayer(Layer):
    
    def __init__(self, n_units, weights, biases):
        Layer.__init__(self, n_units, weights, biases)

    def foward_pass(self, x):
        self.x = x
        y = np.tensordot(self.W, x, (1,0)) + self.b
        return y
    
    def backward_pass(self, dL_dy):
        self.param_gradients(dL_dy)
        return dL_dy.dot( self.W )
        
    def param_gradients(self, dL_dy):      
        self.dL_dW += np.outer(dL_dy.T, self.x.T)
        self.dL_db += dL_dy
        return 
    
class FlattenLayer():
    
    def __init__(self, weights):
        self.weights = weights
        return
    
    def foward_pass(self, x):
        self.y = x.reshape(-1)
        return self.y
    
    def backward_pass(self, dL_dy):
        dL_dx = dL_dy.reshape(*self.weights)
        return dL_dx
    
    def zero_gradients(self, ):
        return

    
class ReLULayer(LinearLayer):
    
    def __init__(self, n_units, weights, biases):
        LinearLayer.__init__(self, n_units, weights, biases)

    def foward_pass(self, x):
        y = super(ReLULayer, self).foward_pass(x)
        self.y = np.maximum(y,0)
        return self.y
    
    def backward_pass(self, dL_dy):
        relu_dL_dy = dL_dy * np.greater(self.y, 0).astype(float)      
        return super(ReLULayer, self).backward_pass(relu_dL_dy)

        
class ConvLayerColumns(Layer):
    
    def __init__(self, depth, filter_size, width, height):
        Layer.__init__(self, depth*width*height, (depth,filter_size*filter_size), biases=True)
        self.stride = 1
        self.padding = int((filter_size - 1)/2) #ensures output size = input size
        self.filter_size = filter_size
        self.depth = depth
        self.width = width
        self.height = height
        #due to parameter sharing the weight vector is
        col_size = filter_size * filter_size
        self.W = (np.random.rand(depth, col_size) - 0.5)/10
        
        
    def pre_pad_all_images(self, X):
        s = X.shape
        N = s[0]
        X_pad = np.zeros((N, self.width + 2*self.padding, self.height + 2*self.padding))
        for n in range(N):
            X_pad[n,:,:] = self.pad_image_reshape(X[n])
            
        return X_pad
            
    def pad_image_reshape(self, x):
        x_r = x.reshape((self.width, self.height))
        return self.pad_image(x_r)
            
    def pad_image(self, x):
        x_pad = np.pad(x, (self.padding, self.padding), 'constant')           
        return x_pad
            
    def im2col(self, X_pad): #convert image to columns one for each receptive field location
        s = X_pad.shape
        N = s[0]
      
        col_size = self.filter_size * self. filter_size
        Cols = np.zeros( (N, self.width, self.height, col_size) )
       
        for n in range(N):
            x = X_pad[n]
            for i in range(self.width):
                for j in range(self.height):
                    col = np.reshape(x[i:i+self.filter_size, j:j+self.filter_size], col_size)
                    Cols[n, i, j] = col

        return Cols            

    def foward_pass(self, x):
        self.x = x # x = w x h x col_size
        y = np.tensordot(self.W, x, axes=(1,2)) # W = depth * col_size
        return y # y = depth * w * h
    
    def backward_pass(self, dL_dy):
        self.param_gradients(dL_dy)
        return dL_dy.dot( self.W )
        
    def param_gradients(self, dL_dy):      
        self.dL_dW += np.outer(dL_dy.T, self.x.T)
        self.dL_db += dL_dy
        return 

    
class ConvLayer(Layer):
    
    def __init__(self, depth, filter_size, width, height):
        Layer.__init__(self, depth*width*height, (depth,filter_size*filter_size), biases=True)
        self.stride = 1
        self.padding = int((filter_size - 1)/2) #ensures output size = input size
        self.filter_size = filter_size
        self.depth = depth
        self.width = width
        self.height = height
        #due to parameter sharing the weight vector is
        self.col_size = filter_size * filter_size    
        self.W = (np.random.rand(depth, self.col_size, depth) - 0.5)/10
        
                            
    def pad_image(self, x):
        x_pad = np.pad(x, (self.padding, self.padding), 'constant')           
        return x_pad
            
    def foward_pass(self, x):
        x = np.pad(x, ((0,0),(self.padding, self.padding),(self.padding, self.padding)), 'constant')           
        self.x = x # x = depth x w x h
        depth = x.shape[0]       
        x_col = np.zeros( (self.width, self.height, depth, self.col_size) ) #14 x 14 x 9

        # rearrange receptive fields into columns
        for i in range(self.width):
            for j in range(self.height):
                x_col[i,j] = np.reshape(x[:, i:i+self.filter_size, j:j+self.filter_size],
                                         (depth, self.col_size))

        y = np.tensordot(self.W, x_col, axes=([0,1],[2,3])) # W =  input depth x col_size x output depth
        return y # y = output depth x width x height
    
    def backward_pass(self, dL_dy):
        self.param_gradients(dL_dy)
        return dL_dy.dot( self.W )
        
    def param_gradients(self, dL_dy):      
        self.dL_dW += np.outer(dL_dy.T, self.x.T)
        self.dL_db += dL_dy
        return 

class MaxPoolLayer(Layer):
    
    def __init__(self, filter_size, width, height, weights, depth):
        Layer.__init__(self, int((width*height)/(filter_size*filter_size)), weights, biases=False)
        self.stride = 2
        self.filter_size = filter_size
        self.width = width #input width
        self.height = height #input height
        self.depth = depth
        
    def foward_pass(self, x):
        self.x = x # x = w x h x depth        
        depth, w, h = x.shape
#         max_pool = x.reshape(depth, int(w/2), 2, int(h/2), 2).max(axis=(2, 4))
        
        max_pool = np.zeros((depth, self.width, self.height))
        f = self.filter_size
        #set weight = 1 for max input to filter at each point
        self.W = np.zeros_like(self.W)
        for d in range(depth):
            for i in range(self.width/self.filter_size):
                for j in range(self.height/self.filter_size):
                    ii = i*f
                    jj = j*f
                    receptive_field = x[d, ii:ii+f, jj:jj+f]
                    max_idx = np.argmax(receptive_field)
                    self.W[max_idx, i,j,d] = 1.0
                    max_pool[d, i, j] = receptive_field.flatten()[max_idx]
                                        
        return max_pool #max_pool = depth x w/2 x h/2
    
    def backward_pass(self, dL_dy):
        return
#         dL_dx = np.zeros((self.depth, self.width, self.height))
#         f = self.filter_size
#         
#         for d in range(self.depth):
#             for i in range(self.width/self.filter_size):
#                 for j in range(self.height/self.filter_size):
#                     ii = i*f
#                     jj = j*f
#                     blah = dL_dy[d,i,j] * self.W[,i,j,d]
#                     dL_dx[d,i,j] = dL_dy[d,i,j] * self.W[,i,j,d]
# 
#         dL_dx = np.tensordot(self.W, dL_dy, (1,0))
#         return dL_dy.dot( self.W )
        
    def param_gradients(self, dL_dy):      
        return     
    
class CrossEntropyLogits():
    
    def __init__(self):
        self.softmax_output = None
        return
    
    def get_loss(self, x, labels):
        exs = np.exp(x)
        sum_exs = np.sum( exs ) 
        self.softmax_output = exs/sum_exs   
        return -np.log(labels.dot(self.softmax_output))
        
    def get_gradient(self, x, labels):              
        return self.softmax_output - labels         
        