# EF-Net: Mental State Recognition by Analyzing Multimodal EEG-fNIRS via CNN
#### Paper: [Sensors 2024](https://doi.org/10.3390/s24061889)

## Abstract

Analysis of brain signals is essential to the study of mental states and various neurological conditions. The two most prevalent noninvasive signals for measuring brain activities are electroencephalography (EEG) and functional near-infrared spectroscopy (fNIRS). EEG, characterized by its higher sampling frequency, captures more temporal features, while fNIRS, with a greater number of channels, provides richer spatial information. Although a few previous studies have explored the use of multimodal deep-learning models to analyze brain activity for both EEG and fNIRS, subject-independent training–testing split analysis remains underexplored. The results of the subject-independent setting directly show the model’s ability on unseen subjects, which is crucial for real-world applications. In this paper, we introduce EF-Net, a new CNN-based multimodal deep-learning model. We evaluate EF-Net on an EEG-fNIRS word generation (WG) dataset on the mental state recognition task, primarily focusing on the subject-independent setting. For completeness, we report results in the subject-dependent and subject-semidependent settings as well. We compare our model with five baseline approaches, including three traditional machine learning methods and two deep learning methods. EF-Net demonstrates superior performance in both accuracy and F1 score, surpassing these baselines. Our model achieves F1 scores of 99.36%, 98.31%, and 65.05% in the subject-dependent, subject-semidependent, and subject-independent settings, respectively, surpassing the best baseline F1 scores by 1.83%, 4.34%, and 2.13% These results highlight EF-Net’s capability to effectively learn and interpret mental states and brain activity across different and unseen subjects.

### Dataset

The Dataset used in our paper is a published open access EEG+fNIRS dataset available [here](http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/). 
This dataset consists of simultaneous measurements of EEG and fNIRS signals from 26 healthy subjects performing a Word Generation or Baseline (Resting) task. 






[Code for other baselines may be provided upon request.]
