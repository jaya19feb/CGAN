import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import load_model
from utils.equalizer import *



def plot_sample_acube(acube,figname):        
        fig=plt.figure()
        fig.suptitle('sample cube.')
        plt.style.use('grayscale')
       
        #for k in range(48):
            #y=fig.add_subplot(6,8,k+1)
            #y.imshow(acube[:,:,k+48*4])
            #print(k+48*4)
            
        for k in range(32):
            y=fig.add_subplot(6,8,k+1)
            #y.imshow(acube[:,:,k+48*2])
            y.imshow(acube[:,:,k])
            
            #print(k+48*4)
                    
        print("calling plt.show()", flush=True)
    
        plt.show()
        
        plt.savefig(figname)
        
        
        


test_unseen="path to .pkl"


print('......', test_unseen)

data_df = pd.read_pickle(test_unseen)
print('data_df', data_df)
data_df=data_df.sample(frac=1)
length_df = len(data_df)
print('....length_df', length_df)
print('....length test_df', len(data_df))



gen_model =load_model("path")


print('Started testing ........')
####make change here for test read the pickle data then extract the data cubes.


print("########## loading test unseen data #############")

data_df = pd.read_pickle('path')



data_df['predicted_cube']=''
data_df['normalized_data_cube']=''
data_df['predicted_cube_normalized']=''
data_df['denormalized_normalized_data_cube']=''
data_df['diff_denorm_orig_and_predicted_denorm_cube']=''

#data_df=data_df.sample(frac=1)
        
print('........data_df', data_df)
print('........data_df len', len(data_df)) 
print('........test_df', data_df)
print('........test_df', len(data_df))
        
data_df=data_df[data_df['augementation_id']=='original_x0']
print('test df few rows ', data_df.head())
print('data df len ',len(data_df))
count=0

#normalize the data to -1 1
instances=np.array(data_df['data_cube'].tolist())
print('instance shape ',instances.shape)
#print(instances)
print("equalizing the data...", flush=True)
eq = histEq(instances)
instances = eq.equalize(instances)
print("normalizing the data...", flush=True)
min_v = np.min(instances)
max_v = np.max(instances)
mean_v = np.mean(instances)
norm_data = np.array([mean_v, min_v, max_v])
instances = (instances - mean_v) / (max_v - min_v)

for index in range(0,len(data_df)):
    data_df.iloc[index, data_df.columns.get_loc('normalized_data_cube')] =instances[index][:][:][:]

print('data_df len *******',len(data_df))
for index, test_row in data_df.iterrows():  # loop over each row of the table, add prediction and then it preserves to get the attached.
    given_cube=test_row['normalized_data_cube']
    #print('LENGTHHHHHH', len(given_cube))
   
    plot_sample_acube(given_cube,'GIVE_CUBE_STEP3')
    
    print('given cube shape', given_cube.shape) #given cube shape (32, 32, 32)
    
    given_cube=given_cube.reshape(-1, 32, 32, 32, 1)
    
    #given_cube=given_cube.reshape(-1, 64, 64, 8, 1)
    
    fake_cube = gen_model.predict([given_cube])
    
    print('<<>><<>><<>><<>>', fake_cube.shape)
    
    #plot_sample_acube(fake_cube,'FAKE_CUBE_STEP3')
    
    #print('.......fake_cube....', fake_cube)
    #fake_cube prediction from generator
    print('.......fake_cube....', fake_cube.shape) #.......fake_cube.... (1, 32, 32, 32, 1)
    fake_cube=fake_cube.reshape(32, 32, 32)
    
    
        
    plot_sample_acube(fake_cube,'FAKE_CUBE_STEP3')
    
    print('given cube normalized ', given_cube)
    print('fake cube data ', fake_cube)
    #if count>2:
    #    exit();

    
    print('count ', count)
    print('inxed ', index)
    
    data_df.iloc[count, data_df.columns.get_loc('predicted_cube_normalized')] =fake_cube
    count=count+1
    #print('.......data_df....', data_df.iloc[index, data_df.columns.get_loc('predicted_cube')])
    print('......given_cube shape', given_cube.shape)#......given_cube shape (1, 32, 32, 32, 1)

    print('........fake_cube shape', fake_cube.shape)#........fake_cube shape (32, 32, 32)

#denormalized
instances_normalized=np.array(data_df['predicted_cube_normalized'].tolist())
print('instance shape ',instances.shape)
#print(instances)
print("de equalizing the data...", flush=True)
#instances_normalized = eq.dequalize(instances_normalized)

print("de normalizing the data...", flush=True)
denormalized_instances=instances_normalized*(max_v - min_v)+ mean_v

print('denormalized instances ',denormalized_instances.shape)


#denormalized of normalization for comparision
orgi_normalized=np.array(data_df['normalized_data_cube'].tolist())
denormalized_orig=orgi_normalized*(max_v - min_v)+ mean_v

for index in range(0,len(data_df)):
    data_df.iloc[index, data_df.columns.get_loc('denormalized_normalized_data_cube')] =denormalized_orig[index][:][:][:]

for index in range(0,len(data_df)):
    data_df.iloc[index, data_df.columns.get_loc('predicted_cube')] =denormalized_instances[index][:][:][:]
    data_df.iloc[index, data_df.columns.get_loc('diff_denorm_orig_and_predicted_denorm_cube')] =denormalized_orig[index][:][:][:]-denormalized_instances[index][:][:][:]
    
    

print('predicted denormalized ', data_df.head())
print('....length of data df', len(data_df))

data_df.to_pickle('path')
print('Done')

