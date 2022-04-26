import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error
import time



def mse(imageB, imageA):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def RMSE(imageB, imageA):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    #err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    #err /= float(imageA.shape[0] * imageA.shape[1])
    err=mean_squared_error(np.array(imageA), np.array(imageB), squared=False)
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imgB, imgA):
    # compute the mean squared error and structural similarity
    # index for the images
    img1 = np.array(imgB).astype(np.uint16)
    img2 = np.array(imgA).astype(np.uint16)
   # img1 = Image.fromarray(imgA, 'L')
   # img2 = Image.fromarray(imgB, 'L')


    #imageAGray = Image.fromarray(ImageOps.grayscale(img1))
    #imageBGray = Image.fromarray(ImageOps.grayscale(img2))

   # imageAGray = ImageOps.grayscale(img1).convert('LA')
   # imageBGray = ImageOps.grayscale(img2).convert('LA')

    m = mse(imgB, imgA)
    r = RMSE(imgB, imgA)
    s = ssim(img2, img1, multichannel=True)
    # setup the figure
    #return plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    return ("MSE: %.2f, RMSE : %.2f, SSIM: %.2f" % (m,r,s))

def pasteCube(X, cube, center, padd=0):  #center is a 3d coord (zyx)
    #print('inside pastecube function')
    #center = center.astype(int)
    #print('center', center)
    #print('center', type(center))
    #center = center.astype(int)
    
    center = list(map(int, center))


    hlz = np.round(cube.shape[0] / 2)
    hlx = np.round(cube.shape[1] / 2)
    hly = np.round(cube.shape[2] / 2)
    #print('X............', X)
    Xn = np.copy(X)
    #print('hlz.........', hlz)
    #print('hlx.........', hlx)
    #print('hly.........', hly)
    #print('Xn.........', Xn)

    #add padding if out of bounds
    if ((center - hlz) < 0).any() or ((center - hlx) < 0).any() or ((center - hly) < 0).any()  or ((center + hlz + 1) > np.array(X.shape)).any() or ((center + hlx + 1) > np.array(X.shape)).any() or ((center + hly + 1) > np.array(X.shape)).any():  # if cropping is out of bounds, add padding
        #print('inside if..........')
        Xn = np.ones(np.array(X.shape) + np.max(cube.shape) * 2) * padd
        Xn[cube.shape[0]:(cube.shape[0] + X.shape[0]), cube.shape[1]:(cube.shape[1] + X.shape[1]), cube.shape[2]:(cube.shape[2] + X.shape[2])] = X
        center = center + np.array(cube.shape)
        Xn[int(center[0] - hlz):int(center[0] - hlz + cube.shape[0]), int(center[1] - hlx):int(center[1] - hlx + cube.shape[1]),
        int(center[2] - hly):int(center[2] - hly + cube.shape[2])] = cube
        Xn = Xn[cube.shape[0]:(Xn.shape[0] - cube.shape[0]), cube.shape[1]:(Xn.shape[1] - cube.shape[1]), cube.shape[2]:(Xn.shape[2] - cube.shape[2])]
    else:
        #print('inside else..........')
        
        Xn[int(center[0] - hlz):int(center[0] - hlz + cube.shape[0]), int(center[1] - hlx):int(center[1] - hlx + cube.shape[1]),
        int(center[2] - hly):int(center[2] - hly + cube.shape[2])] = cube
    #print('Xn............', Xn)
    return Xn

def plot_sample(X, X_fake,X_diff,slice_number, filename, filename_fake, filename_difference):
    
    print('start printing=====')
    #print('X',X)
    #print('XFAKE',X_fake)
    #print('XDIFF',X_diff)
    #print('SLICENUM',slice_number)
    
    
    
    
    fig = plt.figure()
    plt.xlim(0, 512)
    plt.ylim(512, 0)
    # Show some slice in the middle
    plt.imshow(X[slice_number,:,:], cmap=plt.cm.gray)
    #plt.colorbar()
    #plt.show()
    #plt.savefig('denormalizedpredictedcube32.png')
    #plt.savefig('denormalized_normalized_data_cube32.png')
    #plt.savefig('diff_denorm_orig_and_predicted_denorm_cube32.png')
    #plt.savefig('diff_denorm_orig_and_predicted_denorm_cube32_patient18df2.png')
    #plt.savefig('difference34augtesting.png')
    #plt.savefig('patient18_difference_34augtesting_updated_slice56.png')
    plt.savefig(filename)
    plt.close()
    
    fig1 = plt.figure()
    plt.xlim(0, 512)
    plt.ylim(512, 0)
    # Show some slice in the middle
    plt.imshow(X_fake[slice_number,:,:], cmap=plt.cm.gray)
    #plt.colorbar()
    #plt.show()
    #plt.savefig('denormalizedpredictedcube32.png')
    #plt.savefig('denormalized_normalized_data_cube32.png')
    #plt.savefig('diff_denorm_orig_and_predicted_denorm_cube32.png')
    #plt.savefig('diff_denorm_orig_and_predicted_denorm_cube32_patient18df2.png')
    #plt.savefig('difference34augtesting.png')
    #plt.savefig('patient18_difference_34augtesting_updated_slice56.png')
    plt.savefig(filename_fake)
    plt.close()
    
    
    fig2 = plt.figure()
    plt.xlim(0, 512)
    plt.ylim(512, 0)
    # Show some slice in the middle
    plt.imshow(X_diff[slice_number,:,:], cmap=plt.cm.gray)
    #plt.colorbar()
    #plt.show()
    #plt.savefig('denormalizedpredictedcube32.png')
    #plt.savefig('denormalized_normalized_data_cube32.png')
    #plt.savefig('diff_denorm_orig_and_predicted_denorm_cube32.png')
    #plt.savefig('diff_denorm_orig_and_predicted_denorm_cube32_patient18df2.png')
    #plt.savefig('difference34augtesting.png')
    #plt.savefig('patient18_difference_34augtesting_updated_slice56.png')
    plt.savefig(filename_difference)
    plt.close()
    
    compare_images_result = compare_images(X_fake[slice_number,:,:], X[slice_number,:,:])
    print('.......................compare_images_result', compare_images_result)




test_unseen="path"


data_df = pd.read_pickle(test_unseen)
#print('data_df', data_df)
print('dfcolumns',data_df.columns)
test_df=data_df.sample(frac=1)
#print('tessttttt', test_df)
length_df = len(data_df)
#print('....length_df', length_df)
#print('....length test_df', len(test_df))

#initilize a 512x512x Total Z
#patientCube=np.zeros(shape=(512,512,86))
Z=301
X=512
Y=512
#init with 0
scan = np.zeros((Z,X,Y))
fake_scan = np.zeros((Z,X,Y))
diff_scan = np.zeros((Z,X,Y))


#print('.....looping')

#this is loop per patient
for index, test_row in test_df.iterrows(): #for each row
    #print('test_row',test_row)
    #print('inside for loop')
    #print('.....inside test df iterrows')
    #print('test_row....', test_row)
    #print('test_row', test_row)
    #if test_row['patientIds']==['402887-000001']:
    #print('...............test_row....', test_row)
    #if test_row['patientIds']=='402887-000036':
    #if test_row['patientIds']=='402887-000031':
    #if test_row['patientIds']=='402887-000016':
    #if test_row['patientIds']=='402887-000034':
    #if test_row['patientIds']=='402887-000040':
    #if test_row['patientIds']=='402887-000018':
    #if test_row['patientIds']=='0200303-224356-138':
    #if test_row['patientIds']=='20200306-211744-190':
    #print(test_row['patientIds'])
    if test_row['patientIds']== 8:
        
        #print('inside patient id 008')
    #if test_row['patientIds']== '20200303-224356-139':
    #if test_row['patientIds']== '20200229-194032-160':
    #if test_row['patientIds']== '20200315-004604-114':
    #if test_row['patientIds']== '20200306-153518-207':
        #print('.....inside patientIds')
#flip_x, flip_y, shift_xs1, shift_xs2, rotate1, rotate2, rotate3, rotate4, rotate5, rotate6
        if test_row['augementation_id']=='original_x0':    # removed [] format
            #print('.....inside augementation_id')
            
            #print(fake_cube=test_row['data'])

            fake_cube=test_row['data_cube'] 
            #cube=test_row['data_cube']
            #fake_cube=test_row['predicted_cube']
            #fake_cube=test_row['predicted_cube_normalized']
          
            cube=test_row['denormalized_normalized_data_cube']
            diff_cube=test_row['diff_denorm_orig_and_predicted_denorm_cube']
            #print('......fake_cube', fake_cube)
            #print('...fake_cube shape...', fake_cube.shape)

            #fake_cube.reshape(32,32,32)
            # print('FAKE CUBE', fake_cube)
            coord=[]
            
            '''coord[0]=test_row['z_coordinate']  # index out of range so ignoring this
               coord[1]=test_row['x_coordinate']
               coord[2]=test_row['y_coordinate']'''
            
            coord.insert(0, test_row['z_coordinate'] + np.round(fake_cube.shape[0] / 2)) # included everything after +
            coord.insert(1, test_row['y_coordinate'] + np.round(fake_cube.shape[1] / 2))
            coord.insert(2, test_row['x_coordinate'] + np.round(fake_cube.shape[2] / 2))
            

            #print('coord', coord)

            scan = pasteCube(scan, cube, coord)  
            
            fake_scan = pasteCube(fake_scan, fake_cube, coord)
            
            diff_scan = pasteCube(diff_scan, diff_cube, coord)
            
            
            #print('scan........', scan)
            #print('scanmcount ........', len(scan))
        
#time.sleep(300)

for i in range(0,300):
#for i in range(32,40):
    plot_sample(scan,fake_scan,diff_scan,i,'data_cube'+str(i)+'.png','denormalized_normalized_data_cube'+str(i)+'.png','diff_denorm_orig_and_predicted_denorm_cube'+str(i)+'.png')
    
    #plot_sample(scan,fake_scan,diff_scan,185,'predictedlungmask.png','DENORMALIZED_normalized_data_cube_lungmask.png','DIFF_denorm_orig_and_predicted_denorm_cube_lungmask.png')
                        

print('DONE')


