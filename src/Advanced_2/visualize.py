'''
Created on 15 Feb 2017

@author: Dave
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import toimage, imresize, imsave
from scipy.stats import ttest_ind

from matplotlib.pyplot import figure, show, axes, sci
import matplotlib.image as mpimg


def generate_in_paintings(model_probs, nsamples):
    samples = np.random.uniform(size=(nsamples,model_probs.shape[0],model_probs.shape[1])) < model_probs
    return samples.astype('float32')
    
def plot_figures(): 
    
#     plt.imshow(f)    
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    img = mpimg.imread('../_static/stinkbug.png')
    lum_img = img[:,:,0]
    imgplot = plt.imshow(lum_img)
    a.set_title('Before')
#     plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation ='horizontal')
    a=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(lum_img)
#     imgplot.set_clim(0.0,0.7)
    a.set_title('After')
#     plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation='horizontal')
    plt.show()   
    
def add_sub_plot(fig, img, nimages, nsamples, sub_plot_idx):
    a = fig.add_subplot(nimages, nsamples + 1, sub_plot_idx)
    a.axis('off')
    plt.imshow(img)
    return sub_plot_idx + 1

def show_in_paintings(samples, images, file_name):       

    nimages = 10
    image_idxs = np.random.randint(0, len(images), (nimages))
    nsamples = 5  
    w = h = 28
        
#     plot_imgs = np.zeros( (h * len(image_idxs), w * (nsamples+1) ) )
         
    fig = plt.figure()
    sub_plot_idx = 1

    for i, image_idx in enumerate(image_idxs):
        rs = np.reshape(images[image_idx], (w,h))
#         plot_imgs[i*h:i*h+h, 0:w] = rs
        sub_plot_idx = add_sub_plot(fig, rs, nimages, nsamples, sub_plot_idx)
        
        for sample_idx in range(nsamples):
            new_image = images[image_idx]
            new_image[-300:] = samples[sample_idx, image_idx, :]
            rs = np.reshape(new_image, (w,h))
#             x = (sample_idx+1)*w
#             y = i*h
#             plot_imgs[y:y+h, x:x+w] = rs
            sub_plot_idx = add_sub_plot(fig, rs, nimages, nsamples, sub_plot_idx)
    
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=None, bottom=0.02, right=0.36, top=0.98, wspace=0.001, hspace=0.12)
    plt.show()
    plt.savefig('test.png')
        
    combined_images = imresize(plot_imgs,3.0)
    toimage( combined_images ).show()
    imsave( file_name + '.png', combined_images )
    
def get_cross_entropy(images, model_probs, gt, file_name):
    
    nimages = model_probs.shape[0] #100
    npixels = model_probs.shape[1] #300
    nsamples = 10
    
    xent_gt = np.zeros((nimages, npixels), dtype='float32')
    xent_ip = np.zeros((nimages, nsamples, npixels), dtype='float32')
    
    samples = generate_in_paintings(model_probs, nsamples) # 10 x 100 x 300
    
    images = np.squeeze(images, axis=2)
    
    show_in_paintings(samples, images, file_name)
    
    for i in range(nimages):
        for j in range(npixels):
            xent_gt[i,j] -= gt[i,j] * np.log(model_probs[i,j]) + \
                                    (1-gt[i,j]) * np.log(1-model_probs[i,j])
            for k in range(nsamples):
                xent_ip[i,k,j] -= samples[k,i,j] * np.log(model_probs[i,j]) + \
                                        (1-samples[k,i,j]) * np.log(1-model_probs[i,j])
                               
    
    print(i, '-step ground truth Xent', np.mean(xent_gt[:,:i]))
    print(i, '-step in-painting Xent', np.mean(xent_ip[:,0,:i])) #don't average across samples for some reason
    
    for i in [10,28,300]:
        print(i, '-step ground truth Xent', np.mean(xent_gt[:,:i]))
        print(i, '-step in-painting Xent', np.mean(xent_ip[:,:,:i]))
        
        t_stat, p_value = ttest_ind(xent_gt[:,:i], xent_ip[:,:,:i], axis=None)
        print(i, '-step t-test: p=', p_value)
        
