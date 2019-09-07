# -*- coding: utf-8 -*-
import pandas
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import h5py
import numpy as np
import pathlib
import warnings
import shutil
warnings.filterwarnings('ignore')
import os
from pydub import AudioSegment
import cv2
def main():
    
    
    #MERGE SONGS FROM MULTIPLE DIRS TO ONE
    files = os.listdir("C:/Users/belov/Downloads/fma_small")
      
    pathlib.Path(f'C:/Users/belov/Documents/FMA_all_songs').mkdir(parents=True, exist_ok=True)   
    
    for g in files:
        src_files = os.listdir(f'C:/Users/belov/Downloads/fma_small/{g}')
        for file_name in src_files:
            full_file_name = os.path.join(f'C:/Users/belov/Downloads/fma_small/{g}', file_name)
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, f"C:/Users/belov/Documents/FMA_all_songs" )

    #MP3 to WAV CONVERSION:
    src_files = os.listdir(f'C:/Users/belov/Documents/FMA_last_more/')
    for file_name in src_files:
        print(file_name)
        AUDIO_DIR = os.path.join(f'C:/Users/belov/Documents/FMA_last_more/', file_name)
        file_name = file_name[:-4]
        sound = AudioSegment.from_mp3(AUDIO_DIR)
        sound.export(f"C:/Users/belov/Documents/FMA_all_songs_wav/{file_name}.wav", format="wav")
    
    #ACQUIRE LABEL DATA
    df = pandas.read_csv('tracks.csv')
    df.set_index('track_id',inplace=True)
    genre_check = " "
    not_used = "NOTUSED: "
    nans = "N0GENREFOR: "
    i = 0
    k = 0
    labels = " Label: "
    src_files = os.listdir(f'C:/Users/belov/Documents/FMA_all_songs_wav/')
    
    for file_name in src_files:
        file_name1 = file_name[:-4]
        
        if i < 2:
            filename_no_zeros = file_name1[5:]
            try:
                genre = df.loc[filename_no_zeros,'genre_top']
                genre = f"{genre} " 
                
                if genre != "nan ":
                    labels += genre
                
                    file_name = f"{file_name} "
                    nans += file_name
                else:
                    k = k + 1
            except:
                k = k + 1
                pass
        elif i < 3:
            filename_no_zeros = file_name1[4:]
            
            try:
                genre = df.loc[filename_no_zeros,'genre_top']
                genre = f"{genre} "
                
                if genre != "nan ":
                    labels += genre
                
                    file_name = f"{file_name} "
                    nans += file_name
                else:
                    k = k + 1
            except:
                k = k + 1
                pass
            
        elif i < 62:
            filename_no_zeros = file_name1[3:]
            
            try:
                genre = df.loc[filename_no_zeros,'genre_top']
                genre = f"{genre} "
                
                if genre != "nan ":
                    labels += genre
                
                    file_name = f"{file_name} "
                    nans += file_name
                else:
                    k = k + 1
            except:
                k = k + 1
                pass
        elif i < 409:
            filename_no_zeros = file_name1[2:]
            
            try:
                genre = df.loc[filename_no_zeros,'genre_top']
                genre = f"{genre} "
                
                if genre != "nan ":
                    labels += genre
                
                    file_name = f"{file_name} "
                    nans += file_name
                else:
                    k = k + 1
                
            except:
                k = k + 1
                pass
            
            
        elif i < 4506:
            filename_no_zeros = file_name1[1:]
            try:
                genre = df.loc[filename_no_zeros,'genre_top']
                genre_check = f"{genre}"
                genre = f"{genre} "
                
                label = labels.split()
                if (label.count(genre_check) <= 699 ):
                    
                    if genre != "nan ":
                        labels += genre
                    
                        file_name = f"{file_name} "
                        nans += file_name
                    else:
                        k = k + 1
                        file_name = f"{file_name} "
                        not_used += file_name
                else:
                    k = k + 1
                        
                    
            except:
                k = k + 1
                pass
        else:
            filename_no_zeros = file_name1
            
            try:
                genre = df.loc[filename_no_zeros,'genre_top']
                genre_check = f"{genre}"
                genre = f"{genre} "
                
                label = labels.split()
                if (label.count(genre_check) <= 699 ):
                    
                    if genre != "nan ":
                        labels += genre
                    
                        file_name = f"{file_name} "
                        nans += file_name
                    else:
                        k = k + 1
                
                        file_name = f"{file_name} "
                        not_used += file_name
                else:
                    k = k + 1
            except:
                k = k + 1
                pass
        i = i + 1
        
        
    nans = nans.split()
    nans = nans[1:]
    
    not_used = not_used.split()
    not_used = not_used[1:]
    
    labels = labels.split()   
    labels = labels[1:]
    
    print(labels.count('Hip-Hop'))
    print(labels.count('Pop'))
    print(labels.count('International'))
    print(labels.count('Folk'))
    print(labels.count('Instrumental'))
    print(labels.count('Electronic'))
    print(labels.count('Experimental'))
    print(labels.count('Rock'))
    
    
    
    values = array(labels)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    #integer_encoded=list(np.repeat(integer_encoded, 5))
    
    #nans=list(np.repeat(nans, 5))
    
    #SHUFFLE FILENAMES AND LABELS
    integer_encoded, nans = shuffle(integer_encoded, nans)
    
    tr_labels = integer_encoded[0:4200]
   
    val_labels = integer_encoded[4200:4900]
    
    test_labels = integer_encoded[4900:5600]
    
    training_name = 'training_specs.hdf5'
    validation_name = 'validation_specs.hdf5'
    test_name = 'testing_specs.hdf5'
    
    #CREATE EMPTY DATASETS
    tr_dataset = h5py.File(training_name, 'a')
    val_dataset = h5py.File(validation_name, 'a')
    test_dataset = h5py.File(test_name, 'a')
    i = 0
    
    #LOAD SONGS AND EXTRACT FEATURES
    for filename in nans:
        
        print(filename)
        im = cv2.imread(f'C:/Users/belov/Documents/SPECS_FMA/{filename[0:6]}.png',0)
        im = np.array(im)
        im = np.transpose(im,(1,0))
           
        if i < 4200:
               if i == 0:
                   tr_dataset.create_dataset('spectrogram',shape=(4200,432,288),dtype=np.float32)
                   
               tr_dataset['spectrogram'][i] = im
           
        elif i < 4900:
               if i == 4200:
                   val_dataset.create_dataset('spectrogram',shape=(700,432,288),dtype=np.float32)
                 
               val_dataset['spectrogram'][i-4200] = im
           
        else:
               if i == 4900:
                   test_dataset.create_dataset('tempogram',shape=(700,432,288),dtype=np.float32)
                   
               test_dataset['spectrogram'][i-4900] = im
               
        i = i + 1
        print(i)
           
    print(f'Dataset - DONE')
    
    tr_temp = tr_dataset['spectrogram']

    #SAVE LABELS
    tr_dataset.create_dataset('labels_spec_int',data = tr_labels,dtype=np.float32)
    val_dataset.create_dataset('labels_spec_int',data = val_labels ,dtype=np.float32)
    test_dataset.create_dataset('labels_spec_int',data = test_labels,dtype=np.float32)
    
    #NORMALIZE FEATURES: ZERO MEAN AND UNIT VARIANCE
    tr_temp = np.transpose(tr_temp[:],(0,2,1)).reshape(-1,432)
    tr_temp_mean = np.mean(tr_temp,axis=0)
    tr_temp_var = np.var(tr_temp,axis=0)
    tr_temp_sd = np.sqrt(tr_temp_var + 1e-8)
    
    
    tr_dataset['spectrogram'][:] = (tr_dataset['spectrogram'][:] - tr_temp_mean.reshape(1,-1,1))/tr_temp_sd.reshape(1,-1,1)
    
    val_dataset['spectrogram'][:] = (val_dataset['spectrogram'][:] - tr_temp_mean.reshape(1,-1,1))/tr_temp_sd.reshape(1,-1,1)
    
    test_dataset['spectrogram'][:] = (test_dataset['spectrogram'][:] - tr_temp_mean.reshape(1,-1,1))/tr_temp_sd.reshape(1,-1,1)
    
    print(f"DATASET NORMALIZED")
    
    tr_dataset.close()
    val_dataset.close()
    test_dataset.close()
    
if __name__ == '__main__':
    main()