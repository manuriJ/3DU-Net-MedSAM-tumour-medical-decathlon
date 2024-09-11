Introduction

This GitHub repo contain the all the code relevant to implementation of the project 

##### Optimizing Deep Learning Models for 3D Brain Tumour Segmentation: A Comparative study of nnU-Net, U-Net, and MedSAM in Detecting Rare Tumour Regions'
This project implement three foundational architectures, U-Net (Ronneberger, Fischer and Brox, 2015b), nnU-Net (Isensee et al., 2021), and MedSAM (Ma et al., 2023). These frameworks are optimized to improve performance in segmenting three brain tumor regions: edema, non-enhancing core, and enhancing core. 

All the implementation are done using the Tumour data from Medical Segmentation Decathlon(Task 01)

The repository consist of three folders containing implementation scripts for each architechure and can found the README file inside with instruction to initialise the training or inference process

Python Version : Python 3.11 in a Linux-based GPU(NVIDIA RTX A6000) cluster environment
Libraries : 
         
    PyTorch 2.3.1

    TensorFlow 2.4.1 

Note : You can use the create_new_train_val_dataset.py to generate your own choice of train and val combination