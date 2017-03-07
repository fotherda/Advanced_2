'''
Created on 28 Jan 2017

@author: Dave
'''

class ConvergenceTester():
 

    def __init__(self, epsilon, decreasing=False, lookback_window=5):
        self.history = [];
        self.epsilon = epsilon
        if decreasing:
            self.switch = -1
        else:
            self.switch = 1
        self.lookback_window = lookback_window

    
    def has_converged(self, metric):       
        next_score = metric
        converged_count=0
        for previous_score in reversed(self.history[-self.lookback_window:]):      
            diff = (next_score - previous_score) * self.switch #diff +ve if 'improving'
            if diff < self.epsilon: #not improving much
                converged_count += 1
            next_score = previous_score
        
        self.history.append(metric)
  
        converged = False
        if converged_count >= self.lookback_window:
            converged = True
                
        return converged
            
        
    