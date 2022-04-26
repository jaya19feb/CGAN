import pickle
import numpy as np
import os
from utils.equalizer import *
#from configcovid19newconddataframe import *
from configcovid19newconddataframe160patientsFOURCUBES import *
#from PIL import Image
#import cv
from skimage.transform import resize



#dataset_path: path to *.np dataset that has been preprocessed
#normdata_path: path to directory where norm and equalization data is stored ('normalization.np','equalization.np')
#img_res: the shape of the orthotopres (cubes) being processed
class DataLoader():
    def __init__(self, dataset_path, normdata_path, img_res=None):
        self.normdata_path = normdata_path
        if img_res is not None:
            self.img_res = img_res
        else:
            self.img_res = configcovid19newconddataframe160patientsFOURCUBES['cube_size']

        #self.m_xlims = configcovid19newcondCopy1['mask_xlims']
        #self.m_ylims = configcovid19newcondCopy1['mask_ylims']
        #self.m_zlims = configcovid19newcondCopy1['mask_zlims']
        print("loading preprocessed dataset...")
        self.data_train = np.load(dataset_path)
        
        # format for nerual net
        self.data_train = self.data_train.reshape((len(self.data_train), self.img_res[0], self.img_res[1], self.img_res[2], 1))
        # shuffle
        np.random.shuffle(self.data_train)

    def load_data(self, batch_size=1, is_testing=False):
        if is_testing == False:
            idx = np.random.permutation(len(self.data_train))
            batch_images = self.data_train[idx[:batch_size]]
        else:
            idx = np.random.permutation(len(self.data_train))
            batch_images = self.data_train[idx[:batch_size]]
        imgs_A = []
        imgs_B = []
        for i, img in enumerate(batch_images):
            imgs_A.append(img)
            img_out = np.copy(img)
            img_pixelated = np.copy(img)
                #remember to change the reduce size when you change the cube dimension
            for i in range(32):  
               
                    # Resize smoothly down to 16x16 pixels
                    imgSmall = resize(img_pixelated[i], (16, 16))
                    #imgSmall = img_pixelated[i].resize((16,16),Image.BILINEAR)
                    # Scale back up using NEAREST to original size 32 32 in Y and X
                    img_pixelated[i] = resize(imgSmall, (32, 32))
                    #img_pixelated[i] = imgSmall.resize((32,32),Image.NEAREST)
            img_out=img_pixelated
            #img_out[self.m_zlims[0]:self.m_zlims[1], self.m_xlims[0]:self.m_xlims[1], self.m_ylims[0]:self.m_ylims[1]] = 0
            imgs_B.append(img_out)

        return np.array(imgs_A), np.array(imgs_B)

    def load_batch(self, batch_size=1, is_testing=False):
        if is_testing == False:
            self.n_batches = int(len(self.data_train) / batch_size)
        else:
            self.n_batches = int(len(self.data_train) / batch_size)

        for i in range(self.n_batches - 1):
            if is_testing == False:
                batch = self.data_train[i * batch_size:(i + 1) * batch_size]
            else:
                batch = self.data_train[i * batch_size:(i + 1) * batch_size]
            imgs_A = []
            imgs_B = []
            
            for i, img in enumerate(batch):
                imgs_A.append(img)
                #img_out = np.copy(img)
                img_pixelated = np.copy(img)
                #remember to change the reduce size when you change the cube dimension
                for i in range(32):                  
                    # Resize smoothly down to 16x16 pixels
                    #resized_image = cv2.resize(image, (100, 50))
                    
                    imgSmall = resize(img_pixelated[i], (16, 16))
                    
                    
                    #imgSmall = img_pixelated[i].resize((16,16),Image.BILINEAR)
                    # Scale back up using NEAREST to original size 32 32 in Y and X
                    img_pixelated[i] = resize(imgSmall, (32, 32))
                    
                    #imgSmall.resize((32,32),Image.NEAREST)
                img_out=img_pixelated
                #img_out[self.m_zlims[0]:self.m_zlims[1], self.m_xlims[0]:self.m_xlims[1], self.m_ylims[0]:self.m_ylims[1]] = 0
                imgs_B.append(img_out)
            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)
            yield imgs_A, imgs_B

