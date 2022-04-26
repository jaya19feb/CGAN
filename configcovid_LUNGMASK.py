import os
import numpy as np
from keras import backend as K

import tensorflow as tf 

#consider your coordinate system, and x vs y

configcovid_LUNGMASK = {}


configcovid_LUNGMASK['healthy_scans_raw'] = #path to directory where the healthy scans are. Filename is patient ID.


configcovid_LUNGMASK['healthy_coords']= 'path'

#path to csv where each row indicates where a healthy sample is (format: filename, x, y, z). 'fileneame' is the folder containing the dcm files of that scan or the mhd file name, slice is the z axis

configcovid_LUNGMASK['healthy_samples_test1'] =  #path to pickle dump of processed healthy samples for testing.




configcovid_LUNGMASK['healthy_samples_test_array'] = 'path'

configcovid_LUNGMASK['traindata_coordSystem'] = "vox" # the coord system used to note the locations of the evidence ('world' or 'vox'). vox is array index.

configcovid_LUNGMASK['modelpath_inject'] = os.path.join("coviddata","MODELL","INJ") #path to save/load trained models and normalization parameters for injector

configcovid_LUNGMASK['modelpath_remove'] = os.path.join("coviddata","MODELL","REM") #path to save/load trained models and normalization parameters for remover

configcovid_LUNGMASK['progress'] = #path to save snapshots of training progress

# tensorflow configuration
#devices = K.tensorflow_backend._get_available_gpus()
devices = tf.config.list_logical_devices()
if len(devices) > 0: #if there are GPUs avalaible...
    configcovid_LUNGMASK['gpus'] = "0" #sets which GPU to use (use_CPU:"", use_GPU0:"0", etc...)
else:
    configcovid_LUNGMASK['gpus'] = ""

# CT-GAN Configuration
configcovid_LUNGMASK['cube_shape'] = np.array([32,32,32]) #z,y,x
#configcovid19newconddataframe['cube_shape'] = np.array([8,64,64])

configcovid_LUNGMASK['mask_xlims'] = np.array([6,26])
configcovid_LUNGMASK['mask_ylims'] = np.array([6,26])
configcovid_LUNGMASK['mask_zlims'] = np.array([6,26])
configcovid_LUNGMASK['copynoise'] = True #If true, the noise touch-up is copied onto the tampered region from a hardcoded coordinate. If false, gaussain interpolated noise is added instead

# Make save directories
if not os.path.exists(configcovid_LUNGMASK['modelpath_inject']):
    os.makedirs(configcovid_LUNGMASK['modelpath_inject'])
if not os.path.exists(configcovid_LUNGMASK['modelpath_remove']):
    os.makedirs(configcovid_LUNGMASK['modelpath_remove'])
if not os.path.exists(configcovid_LUNGMASK['progress']):
    os.makedirs(configcovid_LUNGMASK['progress'])
