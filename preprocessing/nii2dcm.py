import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.sequence import Sequence
from pydicom.dataset import Dataset

def convert_contour_to_dicom(contour, image, z_position, contour_value=1):
    # Get the indices of the contour pixels
    contour_indices = np.argwhere(contour == contour_value)

    # Append z positions to the 2D indices to get 3D indices
    contour_indices = np.hstack((contour_indices, np.full((contour_indices.shape[0], 1), z_position)))
    
    # Transform from pixel indices to real-world coordinates
    contour_coords = pixel_indices_to_coords(contour_indices, image)
    
    return contour_coords

def pixel_indices_to_coords(pixel_indices, image):
    coords = []
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    for pixel_index in pixel_indices:
        coord = pixel_index * spacing + origin
        coords.append(coord)
    return np.array(coords)


def main(rtstruct_file_path, nii_mask_file_path, new_rtstruct_file_path):
    # Load the RTSTRUCT file
    rtstruct_file = pydicom.dcmread(rtstruct_file_path)

    # Load the binary mask
    mask_img = sitk.ReadImage(nii_mask_file_path)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    print(mask_arr.max())
    # Create new ROI contours for each label
    labels = np.unique(mask_arr)[1:]  # [1:] to skip the background label
    print(labels)
    for label in labels:
        label_mask_arr = np.where(mask_arr == label, 1, 0)
        label_mask_img = sitk.GetImageFromArray(label_mask_arr)
        
        # Find the contours
        contour_filter = sitk.LabelContourImageFilter()
        contour_img = contour_filter.Execute(label_mask_img)
        contour_arr = sitk.GetArrayFromImage(contour_img)

        # Create new contour sequence for this label
        contour_sequence = Sequence()
        
        # For each slice, create a contour and add it to the sequence
        roi_contour = Dataset()
        for i in range(contour_arr.shape[0]):
            contour = contour_arr[i]
            
            # Convert the contour to DICOM format
            dicom_contour = convert_contour_to_dicom(contour, mask_img, z_position=1 )
            #print(type(dicom_contour), dicom_contour.shape)


            # Create a new ROI contour and add it to the sequence
            if len(dicom_contour) > 0:
                print(dicom_contour.shape)
                print(dicom_contour[:10])
                contour_sequence = Sequence()
                roi_contour.ContourData = dicom_contour.flatten('F').tolist() # Flattens the array in column-major order
                contour_sequence.append(roi_contour)
        # # Add the new contour sequence to the RTSTRUCT file
        for dataset in contour_sequence:
            rtstruct_file.ROIContourSequence.append(dataset)   
        # rtstruct_file.ROIContourSequence.append(contour_sequence)

    # Save the modified RTSTRUCT file
    rtstruct_file.save_as(new_rtstruct_file_path)

if __name__ == '__main__':
    rtstruct_file_path = '/mnt/data/shared/hecktor2022/KM_Forskning_01/2021-01__Studies/fsdWvHtaAYEqcuDC_qiZd3pdfha5HHK4r0H15oQA44_RTst_2021-01-28_170228_._Contouring_n1__00000/1.2.246.352.221.489410830223843050315684752710962117256.dcm'
    new_rtstruct_file_path = '/mnt/data/shared/hecktor2022/KM_Forskning_DL_DCM/deeplearning_rtstruct.dcm'
    nii_mask_file_path = '/mnt/data/shared/hecktor2022/KM_Forskning_nii/unetr_pp/KM_Forskning_01.nii.gz'

    main(rtstruct_file_path, nii_mask_file_path, new_rtstruct_file_path)