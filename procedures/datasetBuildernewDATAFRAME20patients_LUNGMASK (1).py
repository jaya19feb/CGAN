from configcovid_LUNGMASK import *
import pandas as pd
from IPython.display import display
import multiprocessing
from scipy.ndimage.interpolation import rotate
from joblib import Parallel, delayed
import itertools
from numpy import load
#from utils.utils import *
from utils.utilsCopy1 import *
#from utils.equalizer import *
from utils.equalizerCopy1 import *
#from utils.dicom_utils import *
from utils.dicom_utils_LUNGMASK import *
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
#for each patient (series) dicom take only 20 slices 
def save_to_dataframe(df, patientIds, x_list, y_list, z_list, instances, instances_ids):
    df = df.append({'patientIds':patientIds, 'x_coordinate':x_list, 'y_coordinate':y_list, 'z_coordinate':z_list, 'data_cube': instances, 'augementation_id':instances_ids}
                   ,ignore_index = True)
    return df

class Extractor:
    #is_healthy_dataset: indicates if the datset are of healthy scans or of unhealthy scans
    #src_dir: Path to directory containing all of the scans (folders of dicom series, or mhd/raw files)
    #dst_path: Path to file to save the dataset in np serialized format. e.g., data/healthy_samples.npy
    #norm_save_dir: the directory where the normalization parameters should be saved. e.g, data/models/INJ/
    #coords_csv: path to csv of the candidate locations with the header:   filename, z, x, y  (if vox, slice should be 0-indexed)
    #   if filename is a directory or has a *.dcm extension, then dicom format is assumed (each scan should have its own directory contianting all of its dcm slices)
    #   if filename has the *.mhd extension, then mhd/raw is assumed (all mdh/raw files should be in same directory)
    #parallelize: inidates whether the processign should be run over multiple CPU cores
    #coordSystem: if the coords are the matrix indexes, then choose 'vox'. If the coords are realworld locations, then choose 'world'
    def __init__(self, is_healthy_dataset, healthy_coords, src_dir=None, coords_csv_path=None, dst_path=None, dst_path_test=None, norm_save_dir=None, parallelize=False, coordSystem=None, test=None, train=None):

        self.parallelize = parallelize
        if coordSystem is None:
            self.coordSystem = configcovid_LUNGMASK['traindata_coordSystem']
        
            
            if healthy_coords =="path":
                self.dst_path_test = dst_path_test if dst_path_test is not None else configcovid_LUNGMASK['healthy_samples_test1']
            if healthy_coords =="path":
                self.dst_path_test = dst_path_test if dst_path_test is not None else configcovid_LUNGMASK['healthy_samples_test2']
            if healthy_coords =="path":
                self.dst_path_test = dst_path_test if dst_path_test is not None else configcovid_LUNGMASK['healthy_samples_test3']
            #if healthy_coords =="path":
                #self.dst_path_test = dst_path_test if dst_path_test is not None else configcovid_LUNGMASK['healthy_samples_test4']
            if healthy_coords =="path":
                self.dst_path_test = dst_path_test if dst_path_test is not None else configcovid_LUNGMASK['healthy_samples_test5'] 
            if healthy_coords =="path":
                self.dst_path_test = dst_path_test if dst_path_test is not None else configcovid_LUNGMASK['healthy_samples_test6']
            if healthy_coords =="path":
                self.dst_path_test = dst_path_test if dst_path_test is not None else configcovid_LUNGMASK['healthy_samples_test7']
            #if healthy_coords =="path":
                #self.dst_path_test = dst_path_test if dst_path_test is not None else configcovid_LUNGMASK['healthy_samples_test8']
            if healthy_coords =="path":
                self.dst_path_test = dst_path_test if dst_path_test is not None else configcovid_LUNGMASK['healthy_samples_test9']
            if healthy_coords =="path":
                self.dst_path_test = dst_path_test if dst_path_test is not None else configcovid_LUNGMASK['healthy_samples_test10']
           

                
            self.norm_save_dir = norm_save_dir if norm_save_dir is not None else configcovid_LUNGMASK['modelpath_remove']
            
        #for coords in configcovid19newconddataframeCopy1['healthy_coords']:
            print('healthy_coords is......', healthy_coords)
            self.coords = pd.read_csv(coords_csv_path) if coords_csv_path is not None else pd.read_csv(healthy_coords)
            #coords has patient_id, X,Y,Z, Path
            
            print('coord......', self.coords)

            print(self.coords.columns)
            
        else:
            self.src_dir = src_dir if src_dir is not None else configcovid_LUNGMASK['unhealthy_scans_raw']
            self.dst_path = dst_path if dst_path is not None else configcovid_LUNGMASK['unhealthy_samples']
            self.norm_save_dir = norm_save_dir if norm_save_dir is not None else configcovid_LUNGMASK['modelpath_inject']
            self.coords = pd.read_csv(coords_csv_path) if coords_csv_path is not None else pd.read_csv(configcovid_LUNGMASK['unhealthy_coords'])

        train_pct_index = int(0.0 * len(self.coords)) # was 0.7
        self.train, self.test = self.coords[:train_pct_index], self.coords[train_pct_index:]
        #self.train = self.test[0:1] # TEMP
        print('train', self.train)
        print('test............', self.test)

    # Train data
    def extract(self, dataType, healthy_coords, plot=True):
        print(".........dataType.......", dataType, flush=True)
        train = self.train
        test = self.test
        print('train in extract', train, flush=True)
        print('test in extract............', test, flush=True)
        if dataType == "train":
            self.coords = train
        if dataType == "test":
            self.coords = test
        # Prep jobs (one per coordinate)
        print("preparing jobs...", flush=True)
        
        J = [] #jobs
        patientIds=[]
        coX=[]
        coY=[]
        coZ=[]
########
        numAugementation=64 #Jaya added it
        #numAugementation=11 #Jaya added it
        #numAugementation=6
        #numAugementation=9
        #numAugementation=16
        #numAugementation=8
        #numAugementation=28
        #numAugementation=19
        #numAugementation=34
       
        column_names ={'patientIds', 'x_coordinate', 'y_coordinate', 'z_coordinate', 'data_cube', 'augementation_id'}        
        df = pd.DataFrame(columns = column_names)
##############  
        print('self.coords.........................................', self.coords)
        for i, sample in self.coords.iterrows():
            print('@@@@@@@@@@@@@@@@@@@@@@@@', i)
            print("printing sample", flush=True)
            print(sample, flush=True)
            print('sample.z, sample.y, sample.x', sample['z'], sample.y, sample.x, flush=True)
            coord = np.array([sample.z, sample.y, sample.x])
            print("coord is", flush=True)
            print(coord, flush=True)
            #input_dicom_folder=sample['dicom_path'] REMOVED AS WE ARE USING .NPY FILES
            #print(input_dicom_folder, flush=True)
            
            if not pd.isnull(sample.z):
                #job: (path to scan, coordinate, instance shape, coord system 'vox' or 'world')
                #J.append([input_dicom_folder, coord,configcovid19newconddataframe160patients['cube_shape'], self.coordSystem])
                J.append([coord,configcovid_LUNGMASK['cube_shape'], self.coordSystem])
                patientIds.append(sample.filename)
                coX.append(sample.x)
                coY.append(sample.y)
                coZ.append(sample.z)

        print("extracting and augmenting samples...", flush=True)
        print('self.parallelize ',self.parallelize, flush=True)
        if self.parallelize:
            num_cores = int(np.ceil(min(np.ceil(multiprocessing.cpu_count() * 0.75), len(J))))
            X = Parallel(n_jobs=num_cores)(delayed(self._processJob)(j) for j in J)
        else:
            X = []
            i=0
            for job in J:
                #try:
                    data, agu_ids=self._processJob(job) #11 cube augementation per co-ordinate
                    print('.........agu_ids', agu_ids, flush=True)
                    #print('$$$$$$$$$$$$$$$$$', data[0].shape)
                    X.append(data)
                    #if len(data)>0:
                        #if (data[0].shape[0]==32) and (data[0].shape[1]==32) and (data[0].shape[2]==32):
                             #X.append(data)
                    
                        
                    
                    
                    for e in range(numAugementation):
                        #print('AAAAAAAAA......i', i)
                        #print('eeeeeeeee......e', e)
                        print('patientIds......', patientIds[i])
                        print('coX[i]......e', coX[i])
                        print('coY[i]......e', coY[i])
                        print('coZ[i]......e', coZ[i])
                        #print('data[e]......e', data[e])
                        #print('agu_ids[e]......e', agu_ids[e])
                        #print('.......', (coY[i] not in [96,128]))                        
                        #print('.......x', coX[i] != 96)
                        #print('.......z', coZ[i] != 352)
                      

                        #print('.......', coX[i] != 96 and (coY[i] not in [96,128]) and coZ[i] != 352)

                        #if coX[i] != 96 and (coY[i] not in [96,128]) and coZ[i] != 352:
                        
                        #if(coX[i] not in [288,320,352,128,384]) and (coY[i] not in [224,192,256,288]) and (coZ[i] not in [64,96,128,224,160]):
                        #if(coX[i] !=288 and coY[i]!=224 and coZ[i] !=64):  
                        #commented this below if statement after adding equalization and normalization
                        #if(coX[i] not in [288,320,352,128,384,256,416]) and (coY[i] not in [224,192,256,288,96,320,352]) and (coZ[i] not in [64,96,128,224,160,0,32]):
                        df=save_to_dataframe(df, patientIds[i], coX[i], coY[i], coZ[i], data[e], agu_ids[e])
                    i=i+1
                    
        print("Done jobs", flush=True)
        print("type of X", type(X))
              
        instances = np.array(list(itertools.chain.from_iterable(X))) #each job creates a batch of augmented instances: so collect them
        #added from here 
        print("equalizing the data...", flush=True)
        eq = histEq(instances)
        instances = eq.equalize(instances)
        os.makedirs(self.norm_save_dir,exist_ok=True)
        
        if dataType=="train":
            eq.save(path=os.path.join(self.norm_save_dir,'equalizationTrainMemory.pkl'))
            
        if dataType=="test":
           
            if healthy_coords =="path":
                eq.save(path=os.path.join(self.norm_save_dir,'equalization1mask.pkl'))
            if healthy_coords =="path":
                eq.save(path=os.path.join(self.norm_save_dir,'equalization2mask.pkl'))
            if healthy_coords =="path":
                eq.save(path=os.path.join(self.norm_save_dir,'equalization3mask.pkl'))
            #if healthy_coords =="path":
                #eq.save(path=os.path.join(self.norm_save_dir,'equalization4mask.pkl'))
            if healthy_coords =="path":
                eq.save(path=os.path.join(self.norm_save_dir,'equalization5mask.pkl'))
            if healthy_coords =="path":
                eq.save(path=os.path.join(self.norm_save_dir,'equalization6mask.pkl'))
            if healthy_coords =="path":
                eq.save(path=os.path.join(self.norm_save_dir,'equalization7mask.pkl'))
            #if healthy_coords =="path":
                #eq.save(path=os.path.join(self.norm_save_dir,'equalization8mask.pkl'))
            if healthy_coords =="path":
                eq.save(path=os.path.join(self.norm_save_dir,'equalization9mask.pkl'))
            if healthy_coords =="path":
                eq.save(path=os.path.join(self.norm_save_dir,'equalization10mask.pkl'))
        #ended here
        #started here
        # -1 1 Normalization
        print("normalizing the data...", flush=True)
        min_v = np.min(instances)
        max_v = np.max(instances)
        mean_v = np.mean(instances)
        norm_data = np.array([mean_v, min_v, max_v])
        instances = (instances - mean_v) / (max_v - min_v)
        #ended here
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #commented down norm_data
        #norm_data=instances
        if dataType=="train":
            np.save(os.path.join(self.norm_save_dir,'normalizationTrainMemory.npy'),norm_data)
        
        if dataType=="test":
           
            if healthy_coords =="path":
                np.save(os.path.join(self.norm_save_dir,'normalization1mask.npy'),norm_data)
            
            
            
            
            
        print('data frame shape ', df.shape, flush=True)

        print('...........shape of instances is', instances.shape, flush=True)
        print('.....length of inst', len(instances), flush=True)
        if dataType=="train":
            #num_df.to_pickle(self.dst_path+"_dataframeTrain.pkl")
            df.to_pickle("path")
        
        if dataType=="test":
           
            if healthy_coords =="path":
                df.to_pickle("path") 
            if healthy_coords =="path":
                df.to_pickle("path")
            if healthy_coords =="path":
                df.to_pickle("path")
            if healthy_coords =="path":
                df.to_pickle("path")
           
            
            
            
            
        if plot:
            print("plotting the samples", flush=True)
            self.plot_sample(instances, dataType, healthy_coords)
            #self.plot_sample_acube(instances[4])
            #a_cube=instances[0]

            #plot it 3d 32x32x32
            '''for i in range(32):
                axial_image = a_cube[:,:,i]
                plt.figure(figsize=(5, 5)) 
                plt.style.use('grayscale') 
                plt.imshow((axial_image))
                plt.savefig('plotsample.png')'''
                

        print("saving the dataset", flush=True)
        #np.save(self.dst_path,instances)
        if dataType=="train":
            np.save(self.dst_path,instances)
        if dataType=="test":
            np.save(self.dst_path_test ,instances)
        

    

    def _processJob(self,args):
        
        #print("Working on job: " + "   "+args[3]+" coord (zyx): ", args[1], flush=True)
        #print("printing instances", flush=True)
        instances, augementationIDs = self._get_instances_from_scan(coord=args[0], cube_shape=args[1], coordSystem=args[2])
        #instances, augementationIDs = self._get_instances_from_scan(coord=args[0], cube_shape=args[1], coordSystem=args[2])
        print("instances", flush=True)
        #print(instances)
        return instances, augementationIDs

    def _get_instances_from_scan(self, coord, cube_shape, coordSystem):
        # load scan data
        #print(scan_path)
        print(coord)
        print(cube_shape)
        print(coordSystem)
        #scan, spacing, orientation, origin, raw_slices = load_scan(scan_path)
        
        #replace this change path in .nii 2 path one for mask 1 for orgirignal
        ###########TODO Jaya need to work with PATHS of 3D data cubes ######################
       
        #arrays = {}

        spacing= 1
        orientation=1
        for f in os.listdir('path'): 
            file= f'path/{f}'
            scan=load(file)
       
        
        
        
            print("after scan.........", flush=True)
            # scale the image
            #scan_resized, resize_factor = scale_scan(scan, spacing)
            # compute sample coords as vox
            if coordSystem == 'world': #convert from world to vox
                print("coord is world", flush=True)
                coord = world2vox(coord,spacing,orientation)
                #coord = world2vox(coord,spacing,orientation,origin) WE do not need this for LUNG MASK
                #LUNG MASK in Y, X, Z
                print(coord, flush=True)
            elif coordSystem != 'vox':
                raise Exception("Coordinate conversion error: you can only select world or vox")
            #coordn = scale_vox_coord(coord, spacing)  # ccord relative to scaled scan

            # extract instances
            X = []
            print("init_cube_shape", flush=True)
            #init_cube_shape = get_scaled_shape(cube_shape + 8, 1/spacing)
            print("clean_cube_unscaled", flush=True)
            #clean_cube_unscaled = cutCube(scan, coord, init_cube_shape, padd=-1000)
            #clean_cube_unscaled = cutCube(scan, coord + cube_shape // 2, init_cube_shape, padd=-1000)
            print("cut coordicate=======plot scan")

            #self.plot_sample_acube(scan,"sample_scan.png")
            clean_cube_unscaled = cutCube(scan, coord + cube_shape // 2, cube_shape, padd=-1000) # change init_cube_shape in line 238         to cube_shape.
            print('6666666666666666666666666', clean_cube_unscaled.shape)

            print("cut coordicate=======",coord[0],coord[1],coord[2])
            #self.plot_sample_acube(clean_cube_unscaled,"sample_cube.png")
            #exit()

            print("after clean_cube_unscaled", flush=True) 
            x, resize_factor = scale_scan(clean_cube_unscaled,spacing)
            # perform data augmentations to generate more instances
            #Xaug, augIDs = self._augmentInstance(x)
            Xaug, augIDs = self._augmentInstance(clean_cube_unscaled)
            # trim the borders to get the actual desired shape
            for xa in Xaug:
                center = np.array(x.shape)//2
                print(xa.shape)
                #X.append(cutCube(xa, center, cube_shape, padd=-1000))  # cut out  augmented cancer without extra boundry
            X = Xaug # changed this one and commented line 249
            print("X in get instances", flush=True)
            #print(X)
            print('number of cube', len(X), flush=True)
            return X, augIDs
    
    def _augmentInstance(self, x0):
        # xy flip
        xf_x = np.flip(x0, 1)
        xf_y = np.flip(x0, 2)
        #xf_xy = np.flip(xf_x, 2)
        # xy shift
        xs1 = scipy.ndimage.shift(x0, (0, 4, 4), mode='constant')
        xs2 = scipy.ndimage.shift(x0, (0, -4, 4), mode='constant')
        #xs3 = scipy.ndimage.shift(x0, (0, 4, -4), mode='constant')
        #xs4 = scipy.ndimage.shift(x0, (0, -4, -4), mode='constant')

        # small rotations
        R = []
        for ang in range(6, 360, 6):
        #for ang in range(15, 360, 15):
        #for ang in range(24, 360, 24):
        #for ang in range(72, 360, 72):
        #for ang in range(30, 360, 30):
        #for ang in range(90, 360, 90):
        #for ang in range(12, 360, 12):
            R.append(rotate(x0, ang, axes=(1, 2), mode='reflect', reshape=False))
        #R.append(rotate(x0, 6, axes=(1, 2), mode='reflect', reshape=False))
        #X = [x0, xf_x, xf_y, xf_xy, xs1, xs2, xs3, xs4] + R
        X = [x0, xf_x, xf_y,xs1, xs2] + R
        
        #IDs=['original_x0', 'flip_x', 'flip_y','shift_xs1', 'shift_xs2', 'R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R12','R13','R14']
        
        IDs=['original_x0', 'flip_x', 'flip_y','shift_xs1', 'shift_xs2', 'R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R12','R13','R14','R15','R16','R17','R18','R19','R20','R21','R22','R23','R24','R25','R26','R27','R28','R29','R30','R31','R32','R33','R34','R35','R36','R37','R38','R39','R40','R41','R42','R43','R44','R45','R46','R47','R48','R49','R50','R51','R52','R53','R54','R55','R56','R57','R58','R59']
        
        #IDs=['original_x0', 'flip_x', 'flip_y','shift_xs1', 'shift_xs2', 'R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R12','R13','R14','R15','R16','R17','R18','R19','R20','R21','R22','R23', 'R24','R25','R26','R27','R28','R29']


        # remove instances which are cropped out of bounds of scan
        Res = []
        for x in X:
            if (x.shape[0] != 0) and (x.shape[1] != 0) and (x.shape[2] != 0):
                Res.append(x)
        return Res, IDs
    '''def plot_sample_acube(self,acube,figname):
        import matplotlib.pyplot as plt
        
        fig=plt.figure()
        fig.suptitle('sample cube.')
       
        for k in range(32):
            y=fig.add_subplot(6,8,k+1)
            y.imshow(acube[:,:,k])
                    
        print("calling plt.show()", flush=True)
        plt.show()
        plt.savefig(figname)'''
            
    def plot_sample(self,X,dataType,healthy_coords):
        import matplotlib.pyplot as plt
        r, c = 3, 10
        batch = X[np.random.permutation(len(X))[:30]]
        fig, axs = plt.subplots(r, c, figsize=np.array([30, 10]) * .5)
        fig.suptitle('Random sample of extracted instances: middle slice shown\nIf target samples are incorrect, consider swapping input target x and y coords.')
        cnt = 0
        for i in range(r):
            for j in range(c):
                if cnt < len(batch):
                    #axs[i, j].imshow(batch[cnt][0:,:,:],cmap='bone')
                    #axs[i, j].imshow(batch[cnt][8, :, :],cmap='bone')
                    axs[i, j].imshow(batch[cnt][:,:,5],cmap='bone')
                    #axs[i, j].imshow(batch[cnt][0,:,:],cmap='bone')
                    axs[i, j].axis('off')
                cnt += 1
        print("calling plt.show()", flush=True)
        plt.show()
        
        if dataType=="train":
            plt.savefig('cubecovid2conddataframetrainMemory.png')

        if dataType=="test":
            if healthy_coords =="path":
                plt.savefig('cubecovid1mask.png')
                
            if healthy_coords =="path":
                plt.savefig('cubecovid2mask.png')
                
            if healthy_coords =="path":
                plt.savefig('cubecovid3mask.png')
                
            #if healthy_coords =="path":
                #plt.savefig('cubecovid4mask.png')
                
            if healthy_coords =="path":
                plt.savefig('cubecovid5mask.png')
                
            if healthy_coords =="path":
                plt.savefig('cubecovid6mask.png')
                
            if healthy_coords =="path":
                plt.savefig('cubecovid7mask.png')
                
            #if healthy_coords =="path":
                #plt.savefig('cubecovid8mask.png')
                
            if healthy_coords =="path":
                plt.savefig('cubecovid9mask.png')
                
            if healthy_coords =="path":
                plt.savefig('cubecovid10mask.png')
                
           
