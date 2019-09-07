# -*- coding: utf-8 -*-
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import h5py
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import cv2

def main():
    
    files = os.listdir('C:/Users/belov/Documents/all_wav_files')
    feature_dir =  os.listdir("C:/Users/belov/.spyder-py3/GTZAN_CHROMS_MORE")
    
    #GENERATING LABELS...
    labels = 'Labels:'
    for filename in files:
        song_label = f" {filename}"
        song_label = song_label[:-10]
        labels += song_label
        
    labels = labels.split()   
    labels = labels[1:]
    values = array(labels)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    integer_encoded=list(np.repeat(integer_encoded, 5))
    integer_encoded, feature_dir = shuffle(integer_encoded, feature_dir)
    
    training_name = 'training_chrom_GTZAN_more.hdf5'
    validation_name = 'validation_chrom_GTZAN_more.hdf5'
    test_name = 'testing_chrom_GTZAN_more.hdf5'
    
    #CREATE EMPTY DATASETS
    tr_dataset = h5py.File(training_name, 'a')
    val_dataset = h5py.File(validation_name, 'a')
    test_dataset = h5py.File(test_name, 'a')
    
    i = 0
    
    tr_labels = integer_encoded[0:4200]
    val_labels = integer_encoded[4200:4600]
    test_labels = integer_encoded[4600:5000]
    
    #FILLING UP THE DATASETS WITH SPECTROGRAMS/TEMPOGRAMS/CHROMAGRAMS...
    for filename in feature_dir:
        
        print(filename)
        im = cv2.imread(f'C:/Users/belov/.spyder-py3/GTZAN_CHROMS_MORE/{filename}',0)
        im = np.array(im)
        im = np.transpose(im,(1,0))
    
        if i < 4200:
               if i == 0:
                   tr_dataset.create_dataset('chromagram',shape=(4200,432,288),dtype=np.float32)
               
               tr_dataset['chromagram'][i] = im
           
        elif i < 4600:
               if i == 4200:
                   val_dataset.create_dataset('chromagram',shape=(400,432,288),dtype=np.float32)
              
               val_dataset['chromagram'][i-4200] = im
           
        else:
               if i == 4600:
                   test_dataset.create_dataset('chromagram',shape=(400,432,288),dtype=np.float32)
                   
               test_dataset['chromagram'][i-4600] = im
               
        i = i + 1
        print(i)
           
    print(f'DATASET - DONE')

    tr_chrom = tr_dataset['chromagram']
    
    #SAVING LABELS...
    tr_dataset.create_dataset('labels_chrom_int',data = tr_labels,dtype=np.float32)
    val_dataset.create_dataset('labels_chrom_int',data = val_labels ,dtype=np.float32)
    test_dataset.create_dataset('labels_chrom_int',data = test_labels,dtype=np.float32)
    
    #NORMALIZING FEATURES: ZERO MEAN AND UNIT VARIANCE...
    tr_chrom = np.transpose(tr_chrom[:],(0,2,1)).reshape(-1,432)
    tr_chrom_mean = np.mean(tr_chrom,axis=0)
    tr_chrom_var = np.var(tr_chrom,axis=0)
    tr_chrom_sd = np.sqrt(tr_chrom_var + 1e-8)
    
    tr_dataset['chromagram'][:] = (tr_dataset['chromagram'][:] - tr_chrom_mean.reshape(1,-1,1))/tr_chrom_sd.reshape(1,-1,1)
    val_dataset['chromagram'][:] = (val_dataset['chromagram'][:] - tr_chrom_mean.reshape(1,-1,1))/tr_chrom_sd.reshape(1,-1,1)
    test_dataset['chromagram'][:] = (test_dataset['chromagram'][:] - tr_chrom_mean.reshape(1,-1,1))/tr_chrom_sd.reshape(1,-1,1)
    
    print(f"DATASET NORMALIZED")
    
    tr_dataset.close()
    val_dataset.close()
    test_dataset.close()
    
if __name__ == '__main__': main()