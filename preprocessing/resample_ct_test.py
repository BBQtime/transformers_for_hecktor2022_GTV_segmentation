
from pathlib import Path
from multiprocessing import Pool
import logging

import click
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os 
import sys
import argparse
import pandas as pd
import numpy as np
import SimpleITK as sitk


def main_resample(arguments):
    """ This command line interface allows to resample NIFTI files within a
        given bounding box contain in BOUNDING_BOXES_FILE. The images are
        resampled with spline interpolation
        of degree 3 and the segmentation are resampled
        by nearest neighbor interpolation.

        INPUT_FOLDER is the path of the folder containing the NIFTI to
        resample.
        OUTPUT_FOLDER is the path of the folder where to store the
        resampled NIFTI files.
        BOUNDING_BOXES_FILE is the path of the .csv file containing the
        bounding boxes of each patient.
    """
    p, input_folder,input_label_folder, output_folder, bb_df = arguments
    resampling=(1, 1, 1)
    output_folder = Path(output_folder)
    input_folder = Path(input_folder)
    input_label_folder = Path(input_label_folder)
    output_folder.mkdir(exist_ok=True)

    print(f'Patient {p} started to resample...')
    sys.stdout.flush()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)
    bb = np.array([
        bb_df.loc[p, 'x1'] - 24, bb_df.loc[p, 'y1'] - 12, bb_df.loc[p, 'z1'] - 48,
        bb_df.loc[p, 'x2'] + 24, bb_df.loc[p, 'y2'] + 36, bb_df.loc[p, 'z2']
    ])
    size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
    ct = sitk.ReadImage(
        str([f for f in input_folder.rglob(p + "__CT*")][0].resolve()))
    # pt = sitk.ReadImage(
    #     str([f for f in input_folder.rglob(p + "__PT*")][0].resolve()))
    gtvt = sitk.ReadImage(
        str([f for f in input_label_folder.rglob(p + "*")][0].resolve()))
    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in size])  # sitk is so stupid
    resampler.SetInterpolator(sitk.sitkBSpline)
    ct = resampler.Execute(ct)
    # pt = resampler.Execute(pt)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    gtvt = resampler.Execute(gtvt)

    print(f'Patient {p} resample completed. ')
    sys.stdout.flush()

    sitk.WriteImage(ct, str(
        (output_folder / (p + "__CT.nii.gz")).resolve()))
    # sitk.WriteImage(pt, str(
    #     (output_folder / (p + "__PT.nii.gz")).resolve()))
    sitk.WriteImage(gtvt,
                    str((output_folder / (p + "__gtv.nii.gz")).resolve()))
    print(f'Patient {p} saved.')
    sys.stdout.flush()

def main_revert(arguments):
    patient, segmentation_path, bounding_boxes_file, original_image_path, output_path = arguments

    # Load the original image
    original_image = sitk.ReadImage(str(original_image_path))

    # Load bounding box information
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')
    bb = np.array([
        bb_df.loc[patient, 'x1'] - 24, bb_df.loc[patient, 'y1'] - 12, bb_df.loc[patient, 'z1'] - 48,
        bb_df.loc[patient, 'x2'] + 24, bb_df.loc[patient, 'y2'] + 36, bb_df.loc[patient, 'z2']
    ])

    # Load the predicted mask image
    predicted_mask = sitk.ReadImage(str(segmentation_path))

    # Resample the predicted mask image to the original spacing
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(original_image.GetSpacing())
    resampler.SetSize(original_image.GetSize())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_predicted_mask = resampler.Execute(predicted_mask)

    # Create a blank image with the same size and spacing as the original image
    blank_image = sitk.Image(original_image.GetSize(), sitk.sitkUInt8)
    blank_image.SetSpacing(original_image.GetSpacing())
    blank_image.SetOrigin(original_image.GetOrigin())

    # Replace the pixels inside the bounding box in the blank image with the resampled segmentation
    start_index = [int((bb[i] - original_image.GetOrigin()[i]) / original_image.GetSpacing()[i]) for i in range(3)]
    end_index = [int((bb[i + 3] - original_image.GetOrigin()[i]) / original_image.GetSpacing()[i]) for i in range(3)]
    for z in range(start_index[2], end_index[2]):
        for y in range(start_index[1], end_index[1]):
            for x in range(start_index[0], end_index[0]):
                blank_image[x, y, z] = resampled_predicted_mask[x - start_index[0], y - start_index[1], z - start_index[2]]

    # Write the reverted segmentation to disk
    sitk.WriteImage(blank_image, str(output_path))
    
def resample_images(input_folder, input_label_folder, output_folder, bounding_boxes_file):
    # Load bounding box information
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')

    for p in bb_df.index:
        # Resample images
        main_resample((p, input_folder, input_label_folder, output_folder, bb_df))

def revert_segmentation(patient, segmentation_path, bounding_boxes_file, original_image_path, output_path):
    # Revert the segmentation
    main_revert((patient, segmentation_path, bounding_boxes_file, original_image_path, output_path))


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resample and revert segmentations.')
    parser.add_argument('--mode', type=str, choices=['resample', 'revert'], help='Mode to run: "resample" or "revert".')
    parser.add_argument('--input_folder', type=str, help='Input folder containing images.')
    parser.add_argument('--input_label_folder', type=str, help='Input folder containing labels.')
    parser.add_argument('--output_folder', type=str, help='Output folder to save resampled images/labels.')
    parser.add_argument('--bounding_boxes_file', type=str, help='CSV file containing bounding boxes.')
    parser.add_argument('--segmentation_path', type=str, help='Path to the predicted segmentation.')
    parser.add_argument('--original_image_path', type=str, help='Path to the original image.')
    parser.add_argument('--output_path', type=str, help='Path to save the reverted segmentation.')

    args = parser.parse_args()

    args.input_folder = '/mnt/data/shared/hecktor2022/KM_Forskning_nii'
    args.input_label_folder = '/mnt/data/shared/hecktor2022/KM_Forskning_nii'
    args.output_folder = '/mnt/data/shared/hecktor2022/KM_Forskning_nii/resampled'
    args.bounding_boxes_file = '/mnt/data/shared/hecktor2022/KM_Forskning_nii/bbox.csv'


    if args.mode == 'resample':
        resample_images(args.input_folder, args.input_label_folder, args.output_folder, args.bounding_boxes_file)
    elif args.mode == 'revert':
        revert_segmentation(args.patient, args.segmentation_path, args.bounding_boxes_file, args.original_image_path, args.output_path)
    
    # for args in list_of_args:
    #     main(args)
    # with Pool(cores) as p:
    #     p.map(main, list_of_args)
    #     #p.starmap(main, list_of_args)
    #main()