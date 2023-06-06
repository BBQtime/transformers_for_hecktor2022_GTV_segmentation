import random
import nibabel as nib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import measure

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

import numpy as np
import nibabel as nib
from rt_utils import RTStructBuilder
def convert(nii_path, dicom_series_path, new_rtstruct_path):
    # Load nii.gz volume
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()

    # Load existing RTStruct
    #rtstruct = RTStructBuilder.create_from(dicom_series_path=dicom_series_path, rt_struct_path= rtstruct_path)
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_series_path)
    # Iterate through unique values in nii.gz file and create corresponding RTStruct ROIs
    for roi_value in np.unique(nii_data)[1:]:  # Skip background
        # Get binary mask for this ROI
        binary_mask = nii_data == roi_value
        print(np.sum(binary_mask))
        # Determine the ROI name based on the value
        if roi_value == 1:
            roi_name = "AI_GTV_T"
        elif roi_value == 2:
            roi_name = "AI_GTV_N"
        else:
            raise ValueError(f"Unexpected ROI value {roi_value} in nii.gz file")

        # Create ROI in the RTStruct
        rtstruct.add_roi(mask=binary_mask, color=[255, 255, 0], name=roi_name)

    # Save new RTStruct
    rtstruct.save(new_rtstruct_path)

if __name__ == '__main__':
    #rtstruct_path = '/mnt/data/shared/hecktor2022/KM_Forskning_01/2021-01__Studies/fsdWvHtaAYEqcuDC_qiZd3pdfha5HHK4r0H15oQA44_RTst_2021-01-28_170228_._Contouring_n1__00000/1.2.246.352.221.489410830223843050315684752710962117256.dcm'
    new_rtstruct_path = '/mnt/data/shared/hecktor2022/KM_Forskning_DL_DCM/KM_Forskning_01_AI_rtstruct.dcm'
    nii_path = '/mnt/data/shared/hecktor2022/KM_Forskning_nii/revert_resample/KM_Forskning_01__DL_gtv.nii.gz'
    dicom_series_path= '/mnt/data/shared/hecktor2022/KM_Forskning_01/2021-01__Studies/fsdWvHtaAYEqcuDC_qiZd3pdfha5HHK4r0H15oQA44_CT_2021-01-28_170228_._._n136__00000/'
    convert(nii_path, dicom_series_path, new_rtstruct_path)

    new_rtstruct_path = '/mnt/data/shared/hecktor2022/KM_Forskning_DL_DCM/KM_Forskning_02_AI_rtstruct.dcm'
    nii_path = '/mnt/data/shared/hecktor2022/KM_Forskning_nii/revert_resample/KM_Forskning_02__DL_gtv.nii.gz'
    dicom_series_path= '/mnt/data/shared/hecktor2022/KM_Forskning_02/2020-07__Studies/YXIeldCqxgYxvEhw_q1BE3YPSYPpjfB39yWaYcR06N_CT_2020-07-07_134341_._._n140__00000/'
    convert(nii_path, dicom_series_path, new_rtstruct_path)

    new_rtstruct_path = '/mnt/data/shared/hecktor2022/KM_Forskning_DL_DCM/KM_Forskning_03_AI_rtstruct.dcm'
    nii_path = '/mnt/data/shared/hecktor2022/KM_Forskning_nii/revert_resample/KM_Forskning_03__DL_gtv.nii.gz'
    dicom_series_path= '/mnt/data/shared/hecktor2022/KM_Forskning_03/2021-01__Studies/mdfOijiVMHfiScYa_81igm3vISlnzEN8hjg8Ydogr9_CT_2021-01-28_174114_._._n151__00000/'
    convert(nii_path, dicom_series_path, new_rtstruct_path)
