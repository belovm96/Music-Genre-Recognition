# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def main():
    names_rep = []
    files = os.listdir('C:/Users/belov/Documents/all_wav_files')
    files=list(np.repeat(files, 5))
    hop_length = 512 
    cmap = plt.get_cmap('inferno')
    j = 0
    i = 0
    for filename in files:
        
            names_rep.append(filename)
            
            if names_rep.count(filename) == 1:
                j = 0
            elif names_rep.count(filename) == 2:
                j = 5
            elif names_rep.count(filename) == 3:
                j = 10
            elif names_rep.count(filename) == 4:
                j = 15
            elif names_rep.count(filename) == 5:
                j = 20
         
            songname = f"C:/Users/belov/Documents/all_wav_files/{filename}"
            
            y, sr = librosa.load(songname, mono=True, sr=22050, offset=j,duration=5)
            #Uncomment to generate tempograms...
            """
            oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                     hop_length=hop_length)
            librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                                x_axis='time', y_axis='tempo')
            """
            #Uncomment to generate spectrograms...
            """
            plt.specgram(y, NFFT=2048, Fs=4096, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB', scale_by_freq=False)
            """
            chroma = librosa.feature.chroma_stft(y=y, sr=sr) #generating chromagrams...
            librosa.display.specshow(chroma)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'C:/Users/belov/.spyder-py3/GTZAN_CHROMS_MORE/{filename[:-3].replace(".", "")}_{j}.png')
            plt.clf()
            i = i + 1
            print(i)
        
if __name__ == '__main__':
    main()