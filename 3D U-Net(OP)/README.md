
This a tensorflow implementation of 3D U-Net. 

Initial code from https://github.com/tamerthamoqa/3D-mri-brain-tumour-image-segmentation-medical-decathlon-tensorflow
is updated to add extra data augmentations, new loss function(Focal loss) and generate evaluation metrics for three tumour region Edma, non-enhancing tumour and Enhancing tumour

#### To train the 3D Unet model here follow the below steps :

Step 01 :

Craete 'train' and 'val' folders each with the 'images' and 'masks' subfolders

Step 02 :

Run Creating separate chunk files from the medical decathlon datasets with a specified image height, width and depth sizes

Step 03:

Run the train_unet_segmentation_multi.py

Step 04:

To generate the visualization use the provided jupyter notebook