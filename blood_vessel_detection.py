# -*- coding: utf-8 -*-
"""
Created on Mon May 22 09:07:50 2023

Blood Vessel Detection

@author: Roya Arian, royaarian101@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import pickle
from scipy import linalg
from skimage.transform import radon
from skimage.restoration import inpaint

## pickle data path
pkl_data_path = str(Path(os.getcwd()).parent) + "\\Data\\pkl_data" # inter your path


# loading data  
file = open(os.path.join(pkl_data_path, "dataset.pkl"), 'rb') # load B-scans
aligned_images = pickle.load(file)

file = open(os.path.join(pkl_data_path, "lable.pkl"), 'rb')  # load the RNFL masks
labels_3_class = pickle.load(file)


inpainted_images = {}


def decom(rank, X):
    h, w, _ = X.shape
    X1 = X.transpose(0,1,2).reshape(h,3*w)
    X2 = X.transpose(1,2,0).reshape(w,3*h)
    U1,_ ,_  = linalg.svd(X1)
    U2,_ ,_ = linalg.svd(X2)
    u1 = U1[:, :rank]
    u2 = U2[:, :rank]
    pu1 = u1.dot(u1.T)
    pu2 = u2.dot(u2.T)
    X = np.tensordot(pu1,X,(0,0))
    X = np.tensordot(pu2,X,(0,1))
    X = X.transpose(1, 0, 2)
    return X


train = 0
for key in aligned_images:
    
    inpainted_images[key] = {}
    
    for key_in in aligned_images[key]:
    
        print(f'train = {train}')
        img_i = aligned_images[key][key_in]
        im = cv2.cvtColor(np.uint8(img_i),cv2.COLOR_GRAY2RGB)
        img = np.clip(decom(5, im)/255, 0, 1)[:,:,0]

        
        #### Finding ILM boundary
        label = np.round(labels_3_class[key][key_in])
        label = np.cumprod(label, axis=0)[0]
        
        label = np.where(label!=0, 1, 0)

            
        img_m = np.clip((img + label), 0, 1)
        # plt.imshow(img_m, cmap='gray')
        
        img_inverse = 1 - img_m
        # plt.imshow(img_inverse, cmap='gray')
        
        image_flip = np.array([list(reversed(row)) for row in img_inverse])
        new_img = np.concatenate((image_flip, img_inverse, image_flip), axis=1)
        # plt.imshow(new_img, cmap='gray')
        
        # Apply gamma correction for contrast inhancement
        gamma = 2.2
        gamma_corrected = np.array(255*new_img ** gamma, dtype = 'uint8')
          
        image = gamma_corrected
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
        
        # ax1.set_title("Original")
        # ax1.imshow(image, cmap=plt.cm.Greys_r)
        
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)
        sinogram = radon(image, theta=theta)
        # dx, dy = 0.5 * 328.0 / max(image.shape), 0.5 / sinogram.shape[0]
        # ax2.set_title("Radon transform\n(Sinogram)")
        # ax2.set_xlabel("Projection angle (deg)")
        # ax2.set_ylabel("Projection position (pixels)")
        # ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
        #             extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
        #             aspect='auto')
        
        # fig.tight_layout()
        # plt.show()
        
        sinogram[:,0] = sinogram[:,0] ** 2
        
        x = np.array(range(3*len(image)))
        y = np.exp((-0.02*x)) 
        
        # plt.plot(x,y)
        # plt.show()
        
        
        sinogram *= y
        
        from skimage.transform import iradon
        
        reconstruction_fbp = iradon(sinogram, theta=theta,  filter_name='cosine')
        imkwargs = dict(vmin=-0.5, vmax=0.5)
        
        # plt.figure()
        # plt.imshow(img, cmap=plt.cm.Greys_r)
        
        
        trend  = []
        column = []
        vessel_label = np.zeros_like(img)
        
        for c in range(len(reconstruction_fbp)):
            trend.append(np.sum(reconstruction_fbp[:,c]))
            column.append(c)
            if c>= 15 and c<len(img)-10:
                if np.sum(reconstruction_fbp[:,c])>=np.quantile(trend, 0.85):
                    vessel_label[:,c] = 1
        
        # plt.figure()
        # plt.imshow(vessel_label, cmap=plt.cm.Greys_r)  
        # plt.plot(column, np.clip(trend, 0, 120))
        
        
        
        # vessel_label = binary_opening(binary_dilation(vessel_label*(1-label), disk(1, dtype=bool)))
        vessel_label = vessel_label*(1-label)
        
        gamma_img = np.array(255*(img_i[:,:,0]/255) ** 1.2, dtype = 'uint8')/255
        
        i = np.clip((gamma_img + (vessel_label))*255, 0, 255)
        image_result = inpaint.inpaint_biharmonic(gamma_img, vessel_label)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 8))
        ax1.imshow(gamma_img, cmap=plt.cm.Greys_r)
        ax1.set_title("Original")

        ax2.imshow(img, cmap=plt.cm.Greys_r)
        ax2.set_title("decomposed image")             
        
        ax3.imshow(i, cmap=plt.cm.Greys_r)
        ax3.set_title("masked image")
        
        ax4.imshow(image_result, cmap=plt.cm.Greys_r)
        ax4.set_title("inpainted")
        
 
        inpainted_images[key][key_in] = image_result

        train += 1
        

################## saving ########################
file = open(os.path.join(pkl_data_path, "dataset_inpainted.pkl"), 'wb')
pickle.dump(inpainted_images, file)


