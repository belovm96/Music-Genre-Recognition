# -*- coding: utf-8 -*-
import pathlib
import shutil
import os

#MERGE ALL SONGS INTO ONE DIR
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()  
pathlib.Path(f'C:/Users/belov/Documents/all_wav_files').mkdir(parents=True, exist_ok=True)   
    
for g in genres:
    src_files = os.listdir(f'C:/Users/belov/Documents/genres/{g}')
    for file_name in src_files:
        full_file_name = os.path.join(f'C:/Users/belov/Documents/genres/{g}', file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, f"C:/Users/belov/Documents/all_wav_files" )