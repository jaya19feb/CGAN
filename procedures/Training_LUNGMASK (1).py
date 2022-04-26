# MIT License
# 
# Copyright (c) 2019 Yisroel Mirsky
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function, division
#from configcovid19newconddataframe import *  # user configuration in config.py
#from configcovid19newconddataframe160patients import *
from configcovid_LUNGMASK import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = configcovid_LUNGMASK['gpus']

#from utils.dataloaderCopy1dataframe import DataLoader
from utils.dataloaderLUNGMASK import DataLoader
from keras.layers import Input, Dropout, Concatenate, Cropping3D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling3D, Conv3D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd

from PIL import Image, ImageOps  

import tensorflow as tf
#import keras.backend.tensorflow_backend as ktf
import tensorflow.keras.backend as ktf

from numpy import loadtxt
from keras.models import load_model
from skimage.measure import compare_ssim as ssim
from keras import backend

#def get_session():
    #gpu_options = tf.GPUOptions(allow_growth=True)
    
    #return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#ktf.set_session(get_session())


class Trainer:
    #def __init__(self,dataType,isInjector=True):
    def __init__(self, healthy_samples,isInjector=True):
        self.isInjector = isInjector
        # Input shape
        cube_shape = configcovid_LUNGMASK['cube_shape']
        self.img_rows = configcovid_LUNGMASK['cube_shape'][1]
        self.img_cols = configcovid_LUNGMASK['cube_shape'][2]
        self.img_depth = configcovid_LUNGMASK['cube_shape'][0]
        self.channels = 1
        self.num_classes = 5
        self.img_shape = (self.img_rows, self.img_cols, self.img_depth, self.channels)
        #self.dataLoaders = []
        self.MAX_EPOCHS=100

        # Configure data loader
        if self.isInjector:
            self.dataset_path = configcovid_LUNGMASK['unhealthy_samples']
            self.modelpath = configcovid_LUNGMASK['modelpath_inject']
        else:
            
            if healthy_samples=='path to .npy files':
                self.dataset_path = configcovid_LUNGMASK['healthy_samples_test1']
            if healthy_samples=='path to .npy files':
                self.dataset_path = configcovid_LUNGMASK['healthy_samples_test2']
            if healthy_samples=='path to .npy files':
                self.dataset_path = configcovid_LUNGMASK['healthy_samples_test3']
            #if healthy_samples=='path to .npy files':
                #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test4']
            if healthy_samples=='path to .npy files':
                self.dataset_path = configcovid_LUNGMASK['healthy_samples_test5']
            if healthy_samples=='path to .npy files':
                self.dataset_path = configcovid_LUNGMASK['healthy_samples_test6']
            if healthy_samples=='path to .npy files':
                self.dataset_path = configcovid_LUNGMASK['healthy_samples_test7']
            #if healthy_samples=='path to .npy files':
                #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test8']
            if healthy_samples=='path to .npy files':
                self.dataset_path = configcovid_LUNGMASK['healthy_samples_test9']
            if healthy_samples=='path to .npy files':
                self.dataset_path = configcovid_LUNGMASK['healthy_samples_test10']
            #if healthy_samples=='path to .npy files':
                #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test11']
            #if healthy_samples=='path to .npy files':
                #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test12']
            #if healthy_samples=='path to .npy files':
                #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test13']
            #if healthy_samples=='path to .npy files':
                #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test14']
            #if healthy_samples=='path to .npy files':
                #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test15']
            #if healthy_samples=='path to .npy files':
                #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test16']
            #if healthy_samples=='path to .npy files':
                #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test17']
            #if healthy_samples=='path to .npy files':
                #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test18']
            #if healthy_samples=='path to .npy files':
                #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test19']
            
           
            self.modelpath = configcovid_LUNGMASK['modelpath_remove']
        
        #dataLoaders=[]
        
        
        #    self.dataLoaders.append(self.dataloader)
            
        #print('dataLoaders..........', dataLoaders)
        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 100
        self.df = 100
        
        #self.gf = 50
        #self.df = 50

        optimizer = Adam(0.0002, 0.5)
        optimizer_G = Adam(0.000001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer_G,
                                   metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator([img_B])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
       

    def build_generator(self):
        """U-Net Generator"""

        def get_crop_shape(target, refer):

            # depth, the 4rth dimension
           # cd = (target.get_shape()[3] - refer.get_shape()[3]).value
            cd = (target.get_shape()[3] - refer.get_shape()[3])

            assert (cd >= 0)
            if cd % 2 != 0:
                cd1, cd2 = int(cd / 2), int(cd / 2) + 1
            else:
                cd1, cd2 = int(cd / 2), int(cd / 2)
            # width, the 3rd dimension
           # cw = (target.get_shape()[2] - refer.get_shape()[2]).value
            cw = (target.get_shape()[2] - refer.get_shape()[2])

            assert (cw >= 0)
            if cw % 2 != 0:
                cw1, cw2 = int(cw / 2), int(cw / 2) + 1
            else:
                cw1, cw2 = int(cw / 2), int(cw / 2)
            # height, the 2nd dimension
           # ch = (target.get_shape()[1] - refer.get_shape()[1]).value
            ch = (target.get_shape()[1] - refer.get_shape()[1])

            assert (ch >= 0)
            if ch % 2 != 0:
                ch1, ch2 = int(ch / 2), int(ch / 2) + 1
            else:
                ch1, ch2 = int(ch / 2), int(ch / 2)

            return (ch1, ch2), (cw1, cw2), (cd1, cd2)

        def conv3d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv3d(layer_input, skip_input, filters, f_size=4, dropout_rate=0.5):
            """Layers used during upsampling"""
            u = UpSampling3D(size=2)(layer_input)
            u = Conv3D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)

            # u = Concatenate()([u, skip_input])
            ch, cw, cd = get_crop_shape(u, skip_input)
            crop_conv4 = Cropping3D(cropping=(ch, cw, cd), data_format="channels_last")(u)
            u = Concatenate()([crop_conv4, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape, name="input_image")

        # Downsampling
        d1 = conv3d(d0, self.gf, bn=False)
        d2 = conv3d(d1, self.gf * 2)
        d3 = conv3d(d2, self.gf * 4)
        d4 = conv3d(d3, self.gf * 8)
        d5 = conv3d(d4, self.gf * 8)
        u3 = deconv3d(d5, d4, self.gf * 8)
        u4 = deconv3d(u3, d3, self.gf * 4)
        u5 = deconv3d(u4, d2, self.gf * 2)
        u6 = deconv3d(u5, d1, self.gf)

        u7 = UpSampling3D(size=2)(u6)
        output_img = Conv3D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(inputs=[d0], outputs=[output_img])

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv3D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        model_input = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(model_input, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv3D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        valid = np.zeros((batch_size,) + self.disc_patch)
        fake = np.ones((batch_size,) + self.disc_patch)
        
        print('Load model ........')
        if epochs!=self.MAX_EPOCHS:  
            self.generator=load_model(self.modelpath+'/G_model_19patientsmask64aug8patientsEQNORM.h5')
            self.discriminator=load_model(self.modelpath+'/D_model_19patientsmask64aug8patientsEQNORM.h5')
            
        print('Started training .......')
        for epoch in range(epochs):
            # save model
            
            count=0
            for healthy_samples in configcovid_LUNGMASK['healthy_samples_test_array']: 
            
                if healthy_samples=='path to .npy files':
                    self.dataset_path = configcovid_LUNGMASK['healthy_samples_test1']
                    self.norm_path= 'path to .npy files'
                if healthy_samples=='path to .npy files:
                    self.dataset_path = configcovid_LUNGMASK['healthy_samples_test2']
                    self.norm_path= 'path to .npy files'
                if healthy_samples=='path to .npy files':
                    self.dataset_path = configcovid_LUNGMASK['healthy_samples_test3']
                    self.norm_path= 'path to .npy files'
                #if healthy_samples=='path to .npy files':
                    #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test4']
                    #self.norm_path= 'path to .npy files'
                if healthy_samples=='path to .npy files':
                    self.dataset_path = configcovid_LUNGMASK['healthy_samples_test5']
                    self.norm_path= 'path to .npy files'
                if healthy_samples=='path to .npy files':
                    self.dataset_path = configcovid_LUNGMASK['healthy_samples_test6']
                    self.norm_path= 'path to .npy files'
                if healthy_samples=='path to .npy files':
                    self.dataset_path = configcovid_LUNGMASK['healthy_samples_test7']
                    self.norm_path= 'path to .npy files'
                #if healthy_samples=='path to .npy files':
                    #self.dataset_path = configcovid_LUNGMASK['healthy_samples_test8']
                    #self.norm_path= 'path to .npy files'
                if healthy_samples=='path to .npy files':
                    self.dataset_path = configcovid_LUNGMASK['healthy_samples_test9']
                    self.norm_path= 'path to .npy files'
                if healthy_samples=='path to .npy files':
                    self.dataset_path = configcovid_LUNGMASK['healthy_samples_test10']
                    self.norm_path= 'path to .npy files'
            
                
                
                self.dataloader = DataLoader(dataset_path=self.dataset_path, normdata_path=self.norm_path,
                                         img_res=(self.img_rows, self.img_cols, self.img_depth))
                
                print('++++++++++++++ data loader +++++++++++++',count)
                print('++++++++++++++ file name +++++++++++++',self.dataset_path)
                print('++++++++++++++  model path +++++++++++++',self.modelpath)
                
                for batch_i, (imgs_A, imgs_B) in enumerate(self.dataloader.load_batch(batch_size)):
                #for batch_i, (imgs_A, imgs_B) in enumerate(self.dataloader.load_batch(batch_size)):
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    # Condition on B and generate a translated version
                    fake_A = self.generator.predict([imgs_B])

                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Train the generators
                    g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                    elapsed_time = datetime.datetime.now() - start_time
                    # Plot the progress
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                                      batch_i,
                                                                                                      self.dataloader.n_batches,
                                                                                                      d_loss[0],
                                                                                                      100 * d_loss[1],
                                                                                                      g_loss[0],
                                                                                                      elapsed_time))

                    # If at save interval => save generated image samples
                    if batch_i % sample_interval == 0:
                        self.show_progress(epoch, batch_i)
                    
                count=count+1
                
                self.generator.save(os.path.join(self.modelpath, "G_model_19patientsmask64aug8patientsEQNORM.h5"))  # creates a HDF5 file
                self.discriminator.save(
                    os.path.join(self.modelpath, "D_model_19patientsmask64aug8patientsEQNORM.h5"))  # creates a HDF5 file 'my_model.h5
                print("Saving Models... at EPOCH............. ", epoch)
                print("REMAINING EPOCHs............. ", epochs-epoch)
        
    def show_progress(self, epoch, batch_i):
        filename = "%d_%d.png" % (epoch, batch_i)
        if self.isInjector:
            savepath = os.path.join(configcovid_LUNGMASK['progress'], "injector")
        else:
            savepath = os.path.join(configcovid_LUNGMASK['progress'], "remover")
        os.makedirs(savepath, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.dataloader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict([imgs_B])
        
        #print('.......................imgs_A', imgs_A)
        for (imgA, imgB) in zip(imgs_A, fake_A): 
            compare_images_result = compare_images(imgB, imgA)
            print('.......................compare_images_result', compare_images_result)
           # imgA.compare_result = compare_images_result
            
        #compare_images_result = compare_images(fake_A, imgs_A)
       # print('images_result.......', compare_images_result)
        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
        
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt].reshape((self.img_depth, self.img_rows, self.img_cols))[int(self.img_depth/2), :, :])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(savepath, filename))
        plt.close()
        
def mse(imageB, imageA):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imgB, imgA):
    # compute the mean squared error and structural similarity
    # index for the images
    img1 = np.array(imgB).astype(np.uint8)
    img2 = np.array(imgA).astype(np.uint8)
   # img1 = Image.fromarray(imgA, 'L')
   # img2 = Image.fromarray(imgB, 'L')


    #imageAGray = Image.fromarray(ImageOps.grayscale(img1))
    #imageBGray = Image.fromarray(ImageOps.grayscale(img2))

   # imageAGray = ImageOps.grayscale(img1).convert('LA')
   # imageBGray = ImageOps.grayscale(img2).convert('LA')

    m = mse(imgB, imgA)
    s = ssim(img2, img1, multichannel=True)
    # setup the figure
    #return plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    return ("MSE: %.2f, SSIM: %.2f" % (m, s))
