import json
import os
import shutil

import numpy as np

data_path = r"Task01_BrainTumour_org"
SEED = 315
SPLIT = 0.85

json_filename = os.path.join(data_path, "dataset.json")

try:
    with open(json_filename, "r") as fp:
        experiment_data = json.load(fp)
except IOError as e:
    raise Exception("File {} doesn't exist. It should be part of the "
                    "Decathlon directory".format(json_filename))

# Print information about the Decathlon experiment data
print("-" * 30)

print("Dataset name:        ", experiment_data["name"])
print("Dataset description: ", experiment_data["description"])
print("Tensor image size:   ", experiment_data["tensorImageSize"])
print("Dataset release:     ", experiment_data["release"])
print("Dataset reference:   ", experiment_data["reference"])
print("Dataset license:     ", experiment_data["licence"])  # sic
print("-" * 30)

"""
Randomize the file list. Then separate into training and
validation lists. We won't use the testing set since we
don't have ground truth masks for this; instead we'll
split the validation set into separate test and validation
sets.
"""


def copy_files_to_new_loc(file_list, outpath):
    for fname in file_list:
        # print(fname)
        shutil.copy2(fname, outpath)


def create_new_trainlst(file_list):
    for fname in file_list:
        f = fname.split('\\')[1]


# Set the random seed so that always get same random mix
np.random.seed(SEED)
numFiles = experiment_data["numTraining"]
idxList = np.arange(numFiles)  # List of file indices
np.random.shuffle(idxList)  # Shuffle the indices to randomize train/test/split

trainIdx = int(np.floor(numFiles * SPLIT))  # index for the end of the training files
trainList = idxList[:trainIdx]
testList = idxList[trainIdx:]

# original split
# otherList = idxList[trainIdx:]
# numOther = len(otherList)
# otherIdx = numOther // 2  # index for the end of the testing files
# validateList = otherList[:otherIdx]
# testList = otherList[otherIdx:]

print("Number of training files   = {}".format(len(trainList)))
# print("Number of validation files = {}".format(len(validateList)))
print("Number of testing files    = {}".format(len(testList)))

''' For Preparing nnunet Training Samples'''

trainFiles = []
trainLabs = []
train_lst = []
for idx in trainList:
    trainFiles.append(os.path.join(data_path, experiment_data["training"][idx]["image"]))
    trainLabs.append(os.path.join(data_path, experiment_data["training"][idx]["label"]))
    train_lst.append({"image": experiment_data["training"][idx]["image"],
                      "label": experiment_data["training"][idx]["label"]})

testFiles = []
testLabs = []
test_lst = []
for idx in testList:
    testFiles.append(os.path.join(data_path, experiment_data["training"][idx]["image"]))
    testLabs.append(os.path.join(data_path, experiment_data["training"][idx]["label"]))
    test_lst.append({"image": experiment_data["training"][idx]["image"],
                      "label": experiment_data["training"][idx]["label"]})

#copy_files_to_new_loc(trainFiles, r"data/images")
# copy_files_to_new_loc(trainLabs,r"data/labelsTr")
# copy_files_to_new_loc(testFiles,r"data/imagesTs")
# copy_files_to_new_loc(testLabs,r"data/labelsTs")
#

#
experiment_data['numTraining'] = len(trainList)
experiment_data['numTest'] = len(testList)
experiment_data['training'] = train_lst
experiment_data['test'] = test_lst


with open('data/dataset.json', 'w') as fp:
    json.dump(experiment_data, fp)
