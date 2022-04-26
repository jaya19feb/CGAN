import numpy as np
import pandas as pd
import os
import pydicom  
import scipy
import scipy.misc
import pandas
import math
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
from numpy import load
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
                    
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import nibabel as nib

def display_views(file_path):
    image_axis = 2
    n1_img = nib.load(file_path)
    image = n1_img.get_data()
    
    
    
    for i in range(301):
        axial_image = image[:, :, i] # Axis 2


    #plt.figure(figsize=(20, 10))
    #plt.style.use('grayscale')


    #plt.subplot(142)
    #plt.imshow(np.rot90(axial_image))
    #print(image)
    
    return image

def chunkify(img, block_width=32, block_height=32, length_slices=32):
    print('img.........', img)

    rows=img.shape[0]    #y
    columns=img.shape[1] #x
    #img.shape[0] 
    #total_z=numberOfSlices(dirName)
    total_z=img.shape[2] #z
    
    
    
    #rows=img.X.values[0]
    #columns=img.Y.values[0]
    #total_z=numberOfSlices(dirName)
    #total_z=numberOfSlices(subdir)
    
    
    print('length of slices', length_slices)    
    print('rows is ', rows)
    print('columns is ', columns)
 
    
    shape = [rows, columns,length_slices]
    print("shape", shape)
    #x_len = int(rows/block_width) - 2 
    #y_len = int(columns/block_height) - 1
    
    x_len = int(rows/block_width) 
    y_len = int(columns/block_height) 
    z_len = int(total_z/length_slices)
    
    
    print("number of x_len box ", x_len)
    print("number of y_len box ", y_len)
    print("number of z_len box ", z_len)

    chunks = []
    co_ordinate=[]
    for z in range(0, z_len*length_slices, length_slices): 
        for y in range(0, y_len*block_width, block_height): #15*32= 480
            for x in range(0, x_len*block_width, block_width):#14*32= 448
                '''
                start_box=[0,0,0]
                start_box[0] = x
                start_box[1] = y
                start_box[2] = z
                co_ordinate.append(start_box)
                '''
                start_box=[0,0,0,0]
                start_box[0]= '29864901'
                start_box[1] = x
                start_box[2] = y
                start_box[3] = z
                
                #count_background=np.count_nonzero(result[x:block_width,y:block_height,z:length_slices])
                count_background=np.count_nonzero(data[x:x+block_width,y:y+block_height,z:z+length_slices])
                if (count_background>0):
                    print(count_background)
                    print('you keep this cube')
                    co_ordinate.append(start_box)
                    
    print('finalShapes',co_ordinate)
 
    return co_ordinate

#slices_needed = 20
#column_names = ["PatientID" ,"X", "Y"]
#patientId = ds.PatientID

#print(df.X)
#print('data frame shape ', df.shape)

#readin 3d

#data = load('radiopaedia_29_86490_1.npy')
data = load('fullimages_0.npy')
#img_path = display_views(path)
#print(img_path.shape)
#mask_path =  display_views(path)
#print(mask_path.shape)
#result = img_path*mask_path
#plt.imshow(np.rot90(result[:,:,180]))
#print('result 180')
#print(np.min(result[:,:,180]))
#print(np.max(result[:,:,180]))

#print(np.max(result[10,10,180]))
#filter

#blocks = chunkify(df)
#blocks = chunkify(result)
blocks = chunkify(data)
#blocks = chunkify(data)
df = pd.DataFrame(blocks)
print('df................', df)
df.reset_index()
print('................', df)

#df.to_csv('radiopaedia_29_86490_1.csv', index=False)
df.to_csv('kaggle1.csv', index=False)
      
