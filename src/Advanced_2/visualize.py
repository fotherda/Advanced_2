'''
Created on 15 Feb 2017

@author: Dave
'''
import numpy as np
from scipy.misc import toimage, imresize
from scipy.stats import ttest_ind

def generate_in_paintings(model_probs, nsamples):
#     samples = np.zeros((nsamples, model_probs.shape[0], model_probs.shape[1]))
    samples = np.random.uniform(size=(nsamples,model_probs.shape[0],model_probs.shape[1])) < model_probs
    return samples.astype('float32')
    
def show_in_paintings(samples, images):       

#     image_idxs = range(5)  
    nimages = 20
    image_idxs = np.random.randint(0,len(images), (nimages))
    nsamples = 5  
    w = h = 28
        
    plot_imgs = np.zeros( (h * len(image_idxs), w * (nsamples+1) ) )
         
    for i, image_idx in enumerate(image_idxs):
        plot_imgs[i*h:i*h+h, 0:w] = np.reshape(images[image_idx], (w,h))
        for sample_idx in range(nsamples):
            new_image = images[image_idx]
            new_image[-300:] = samples[sample_idx, image_idx, :]
            rs = np.reshape(new_image, (w,h))
            x = (sample_idx+1)*w
            y = i*h
            plot_imgs[y:y+h, x:x+w] = rs
        
    toimage( imresize(plot_imgs,3.0) ).show()
    
def get_cross_entropy(images, model_probs, gt):
    
    nimages = model_probs.shape[0] #100
    npixels = model_probs.shape[1] #300
    nsamples = 10
    
    xent_gt = np.zeros((nimages, nsamples, npixels), dtype='float32')
    xent_ip = np.zeros((nimages, nsamples, npixels), dtype='float32')
    
    samples = generate_in_paintings(model_probs, nsamples) # 10 x 100 x 300
    
    
    images = np.squeeze(images, axis=2)
    show_in_paintings(samples, images)
    
#     model_preds = np.greater(model_probs, 0.5).astype(int)
#     nz = np.count_nonzero(model_preds, axis=1)
#     fraction = nz / model_preds.size

    
    for i in range(nimages):
        for j in range(npixels):
            xent_gt[i,0,j] -= gt[i,j] * np.log(model_probs[i,j]) + \
                                    (1-gt[i,j]) * np.log(1-model_probs[i,j])
            for k in range(nsamples):
                xent_ip[i,k,j] -= samples[k,i,j] * np.log(model_probs[i,j]) + \
                                        (1-samples[k,i,j]) * np.log(1-model_probs[i,j])
                               
#     xent_gt = np.divide(xent_gt, nimages)
#     xent_ip = np.divide(xent_ip, nimages*nsamples)

    for i in [1,10,28,300]:
        print(i, '-step ground truth Xent', np.mean(xent_gt[:,:,:i]))
        print(i, '-step in-painting Xent', np.mean(xent_ip[:,:,:i]))
        
        t_stat, p_value = ttest_ind(xent_gt[:,:,:i], xent_ip[:,:,:i])
        print(i, '-step t-test: p=', p_value)
        
