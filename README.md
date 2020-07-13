# Music Genre Recognition Research Project
In this project, I proposed a Convolutional Neural Network architecture trained on the data from GTZAN & FMA datasets that achieves 70% accuracy recognizing 10 music genres.

## Motivation & Objective
Automated Music Genre Recognition has been a challenging and interesting problem in the field of Music Information Retrieval that is yet to be resolved. 
Having read a variety of papers that present their approaches to this MIR task, it was apparent that there is a potential for an improvement of the current algorithms deployed for this classification problem. 
Moreover, most of classification results presented in these papers are neither impressive nor compelling which served as a motivation for contributing to this area of MIR. 

With Deep Learning emerging as one of the most popular and promising Machine Learning techniques, it was decided to use Deep Neural Networks for the implementation of Music Genre Recognition algorithm. 

In this project, a novel Convolutional Neural Network Architecture is proposed, and its performance is evaluated on three audio music features: tempogram, chroma frequencies, and spectrogram. 
The goals of this research project are to determine which of the three features yields best classification performance on the proposed CNN, and to provide a discussion on the reasons why certain music attributes are more meaningful for discerning music genres than others.

## Project Recap
* Downloaded GTZAN and FMA (small) datasets
* Generated spectrogram, tempogram and chroma audio features from the 30-second song snippets presented in the two datasets
* Performed feature engineering on the extracted features to capture music genre discepancies
* Built a Convolutional Neural Network of the architecture described in the project report and trained it on the extracted features
* Analyzed the performance of the proposed CNN architecture using accuracy and confusion matrix metrics, and provided a discussion on the project's outcomes and potential improvements

## Frameworks & Tools
* **Languages:** Python 3.6
* **Libraries:** PyTorch, LibROSA, Pandas, NumPy, H5Py, Scikit-learn, Matplotlib

