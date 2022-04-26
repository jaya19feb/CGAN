import numpy as np
import pickle

class histEq:
    def __init__(self,X_sample,levels=10000,nbins=10000, path=[]): #X_sample is a list of one or more images sampled from the distribution, levels is the ourput norm range
        if len(path) > 0:
            self.load(path)
        else:
            self.nbins = nbins
            self.init(X_sample,levels,nbins)
            print('1stX_SAMPLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', X_sample)

    def init(self,X_sample,levels,nbins):
        #format X
        print('SHAPE OF X_SAMPLE#########################', X_sample.shape)
        X_f = np.concatenate(X_sample)
        print('2ndX_SAMPLE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', X_sample)
        print('X_f!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', X_f)
        self.N =  np.prod(X_f.shape) #number of pixels
        self.L = levels #grey levels to eq to

        hist, bins = np.histogram(X_f.ravel(), nbins)
        self.hist = hist/len(X_sample) #normalize to thenumber of images
        self.bins = bins + ((max(bins)-min(bins))/len(bins))/2 #center of each bin
        self.bins = self.bins[:nbins]
        self.cdf = np.cumsum(self.hist)
        self.cdf_min = np.min(self.cdf)

    def equalize(self,image):
        print('.....image', type(image))
        print('.....image', image)
        print('.....image############', image[0].shape)
        print('.....image@@@@@@@@@@@@', type(image[0]))
        print('.....image shape', image.shape)
        print('.....image', len(image)) 
        print('.....image.flat', type(image.flat)) 
        print('.....self.bins', type(self.bins))
        print('.....BINS SHAPE', self.bins.shape)
        print('.....CDF SHAPE', self.cdf.shape)
        print('.....self.cdf', type(self.cdf)) 
        #image = image.astype(np.float64)
        #right now its 3d integer32........> convert to 3d float64
        out=[]
        for i in range(len(image)):
            tmp = np.interp(image[i], self.bins, self.cdf)
            out.append(tmp)
        #out= np.interp(image, self.bins, self.cdf)
        out=np.array(out)
        print('.....out', type(out))
        print('.....out len', len(out))
        print('.....data ############', out[0].shape)
        print('.....data #',out[0])
        
        return out.reshape(image.shape)

    def dequalize(self,image):
        out = np.interp(image.flat, self.cdf, self.bins)
        return out.reshape(image.shape)

    def save(self,path=None):
        if path is None:
            return [self.bins,self.cdf]
        with open(path, 'wb') as f:
            pickle.dump([self.bins,self.cdf], f)

    def load(self,path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.bins = data[0]
        self.cdf = data[1]

