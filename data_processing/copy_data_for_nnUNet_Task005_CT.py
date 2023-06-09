#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
import shutil
import glob
import os

if __name__ == "__main__":
    """
    Copy images and labels into nnunet folders
    """

    task_name = "Task005_CT"
    downloaded_data_dir = "/mnt/data/shared/hecktor2022/train/hecktor2022_training/hecktor2022/resampled/"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    #target_imagesVal = join(target_base, "imagesVal")
    # target_imagesTs = join(target_base, "imagesTs")
    # target_labelsTs = join(target_base, "labelsTs")


    maybe_mkdir_p(target_imagesTr)
    #maybe_mkdir_p(target_imagesVal)
    # maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)
    # maybe_mkdir_p(target_labelsTs)


    patient_names = []
    for path in glob.glob(os.path.join(downloaded_data_dir, '*.nii.gz')):
        file = os.path.basename(path)
        patient_names.append(file.split('__')[0])
    patient_names = sorted(list(set(patient_names)))

    cur = downloaded_data_dir
    #for p in subdirs(cur, join=False):
    for p in patient_names:

        #patdir = join(cur, p)
        patient_name = p

        ct = join(cur, p+"__CT.nii.gz")
        gtv = join(cur, p+"__gtv.nii.gz")

        print(ct)
        assert all([
            isfile(ct),
            # isfile(pt),
            isfile(gtv)
        ]), "%s" % patient_name

        shutil.copy(ct, join(target_imagesTr, patient_name + "_0000.nii.gz"))
        # shutil.copy(pt, join(target_imagesTr, patient_name + "_0001.nii.gz"))
        shutil.copy(gtv, join(target_labelsTr, patient_name + ".nii.gz"))
    print("!Copy finished!")
    json_dict = OrderedDict()
    json_dict['name'] = "hecktor_CT"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "NA"
    json_dict['licence'] = "NA"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "GTVt",
        "2":"GTVn"
    }
    json_dict['numTraining'] = len(patient_names)
    #json_dict['numTest'] = len(test_patient_names)
    json_dict['test'] = []
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_names]


    save_json(json_dict, join(target_base, "dataset.json"))