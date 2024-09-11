The traning process was directly run from the Linux commanline using the follwoing commands

Run the following commands in order as they line to train the nnU-Net

Note : Before executing any command, first create 4 folders in a prefered location to save the data files generated during the training process.

Folder structure is :

    RawData - Save the raw data files here to start the training process

    nnUNet_raw - Converted data is to be saved here

    nnUNet_preprocessed - Preprocessed data and configuration plan will be saved here.

    nnUNet_results - All the results will be saved here( under a folder name reflecting the configuration and no of epochs)



### 6.2.1 Dataset Conversion
nnUNetv2_convert_MSD_dataset -i "${RAW_DATA_PATH}/Task01_BrainTumour"

## 6.2.2 Extracting Rule Based Parameters
nnUNetv2_plan_and_preprocess -d 1

for help nnUNetv2_plan_and_preprocess -h

## 6.2.3 Training nnU-Net
#### to use nnUNetv2_find_best_configuration later, add the --npz flag  here
#### This makes nnU-Net save the softmax outputs during the final validation.

#### --- 2d data  for one epoch ----> change the epochs as defined in the doc
#### 0th fold ---> can train fro all 5 fold seperately and predict later with all five
#### -device cpu - to run with the cpu

##### nnUNetv2_train 1 2d 0 -tr nnUNetTrainer_1epoch
nnUNetv2_train 1 2d 0 -tr nnUNetTrainer_20epochs

#### train a 3D nnU-Net on Full Resolution for 2 epochs.
nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainer_1epoch

## 6.2.4 Predict for the unseen data

####  * provide test dataset path
####  * folder path to save the predicted data ---> pred_TS001
#### add -f all to use data from all five folds


###  make the new folder for predicted data
cd data/nnUNet_results/Dataset001_BrainTumour
mkdir pred_2D_001

nnUNetv2_predict -i "${nnUNet_raw}/Dataset001_BrainTumour/imagesTs" -o "${nnUNet_results}/Dataset001_BrainTumour/pred_2D_001" -d 1 -f 0 -tr nnUNetTrainer_20epochs -c 2d -p nnUNetPlans

nnUNetv2_predict -i "${nnUNet_raw}/Dataset001_BrainTumour/imagesTs" -o "${nnUNet_results}/Dataset001_BrainTumour/pred_3D_001" -d 1 -f 0 -tr nnUNetTrainer_20epochs -c 3d_fullres -p nnUNetPlans

## 6.2.5 Evalute the model

#### 1 ---> give the ground truth folder path
#### 2 -----> predicted folder path eg: pred_TS001

nnUNetv2_evaluate_folder "${RAW_DATA_PATH}/Task01_BrainTumour/labelsTs" "${nnUNet_results}/Dataset001_BrainTumour/pred_2D_001" -djfile "${nnUNet_results}/Dataset001_BrainTumour/nnUNetTrainer_20epochs__nnUNetPlans__2d/dataset.json" -pfile "${nnUNet_results}/Dataset001_BrainTumour/nnUNetTrainer_20epochs__nnUNetPlans__2d/plans.json"

nnUNetv2_evaluate_folder "${RAW_DATA_PATH}/Task01_BrainTumour/labelsTs" "${nnUNet_results}/Dataset001_BrainTumour/pred_3D_001" -djfile "${nnUNet_results}/Dataset001_BrainTumour/nnUNetTrainer_20epochs__nnUNetPlans__3d_fullres/dataset.json" -pfile "${nnUNet_results}/Dataset001_BrainTumour/nnUNetTrainer_20epochs__nnUNetPlans__3d_fullres/plans.json"


#### Optinal steps

#### best model among the trained models ---> neede to have train the models with -npz opt enable
#### will create the postprocessing.pkl
nnUNetv2_find_best_configuration 1

#### post process the predicted results
nnUNetv2_apply_postprocessing -i "${nnUNet_results}/Dataset001_BrainTumour/pred_Ts001" -o "${nnUNet_results}/Dataset001_BrainTumour/Ts001_BrainTumour/postprocessing" -pp_pkl_file "${nnUNet_results}/Dataset001_BrainTumour/nnUNetTrainer_1epoch__nnUNetPlans__2d/folds_0/postprocessing.pkl" -np 8 -plans_json "${nnUNet_results}/Dataset001_BrainTumour/nnUNetTrainer_1epoch__nnUNetPlans__2d/plans.json"

