'''
Created on 31 Jan 2017

@author: Dave
'''

class LearningRateScheduler():
    
    def __init__(self, decay):
        self.decay = decay
        return
    
    def get_learning_rate(self, epoch, learning_rate):
        return learning_rate / (1+ self.decay * epoch)

