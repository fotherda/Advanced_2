'''
Created on 27 Jan 2017

@author: Dave
'''

class Network():
 
    def __init__(self):
        self.layers = []
        
    def addLayer(self, layer):
        self.layers.append(layer)
        
    def addLossFunction(self, loss_function):
        self.loss_function = loss_function       
    
    def forward_pass(self, x):
        for layer in self.layers:
            y = layer.foward_pass(x)
            x = y
        return y        
        
    def backward_pass(self, dL_dy):
        for layer in reversed(self.layers):
            dL_dx = layer.backward_pass(dL_dy)
            dL_dy = dL_dx           
        return dL_dy
        
    def update_parameters(self, learning_rate):
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def zero_gradients(self):
        for layer in self.layers:
            layer.zero_gradients()
        
    def run_one_train_epoch(self, x_batch, labels_batch, learning_rate):
        L = 0.0       
        self.zero_gradients()
        
        for x, labels in zip(x_batch, labels_batch):
            y = self.forward_pass(x) 
            L += self.loss_function.get_loss(y, labels)
            dL_dy = self.loss_function.get_gradient(y, labels)
            self.backward_pass(dL_dy)
                   
        self.update_parameters(learning_rate) 
          
        return L
    
    def get_accuracy(self, x_batch, labels_batch):
        correct = 0
        
        for x, labels in zip(x_batch, labels_batch):
            y = self.forward_pass(x) 
            if y.argmax() == labels.argmax():
                correct += 1
            
        return correct / len(x_batch)    
        
    def get_predictions(self, x_batch):
        
        y_list = []
        for x in x_batch:
            y = self.forward_pass(x) 
            y_list.append(y.argmax())
        
        return y_list    
        
        
        
        