"""
Script for preprocessing the LIDC-IDRI dataset.
"""
import argparse
import os
import shutil
import dicom2nifti
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def preprocess_nifti(input_path, output_path):
    # Load the Nifti image
    print('Process image: {}'.format(input_path))
    img = nib.load(input_path)

    # Get the current voxel sizes
    voxel_sizes = img.header.get_zooms()

    # Calculate the target voxel size (1mm x 1mm x 1mm)
    target_voxel_size = (1.0, 1.0, 1.0)

    # Calculate the resampling factor
    zoom_factors = [current / target for target, current in zip(target_voxel_size, voxel_sizes)]

    # Resample the image
    print("[1] Resample the image ...")
    resampled_data = zoom(img.get_fdata(), zoom_factors, order=3, mode='nearest')

    print("[2] Center crop the image ...")
    crop_size = (256, 256, 256)
    depth, height, width = resampled_data.shape

    d_start = (depth - crop_size[0]) // 2
    h_start = (height - crop_size[1]) // 2
    w_start = (width - crop_size[2]) // 2
    cropped_arr = resampled_data[d_start:d_start + crop_size[0], h_start:h_start + crop_size[1], w_start:w_start + crop_size[2]]

    print("[3] Clip all values below -1000 ...")
    cropped_arr[cropped_arr < -1000] = -1000

    print("[4] Clip the upper quantile (0.999) to remove outliers ...")
    out_clipped = np.clip(cropped_arr, -1000, np.quantile(cropped_arr, 0.999))

    print("[5] Normalize the image ...")
    out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))

    assert out_normalized.shape == (256, 256, 256), "The output shape should be (320,320,320)"

    print("[6] FINAL REPORT: Min value: {}, Max value: {}, Shape: {}".format(out_normalized.min(),
                                                                             out_normalized.max(),
                                                                             out_normalized.shape))
    print("-------------------------------------------------------------------------------")
    # Save the resampled image
    resampled_img = nib.Nifti1Image(out_normalized, np.eye(4))
    nib.save(resampled_img, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_dir', type=str, required=True,
                        help='Directory containing the original dicom data')
    parser.add_argument('--nifti_dir', type=str, required=True,
                        help='Directory to store the processed nifti files')
    parser.add_argument('--delete_unprocessed', type=eval, default=True,
                        help='Set true to delete the unprocessed nifti files')
    args = parser.parse_args()

    # Convert DICOM to nifti
    for patient in os.listdir(args.dicom_dir):
        print('Convert {} to nifti'.format(patient))
        if not os.path.exists(os.path.join(args.nifti_dir, patient)):
            os.makedirs(os.path.join(args.nifti_dir, patient))
        dicom2nifti.convert_directory(os.path.join(args.dicom_dir, patient),
                                        os.path.join(args.nifti_dir, patient))
        shutil.rmtree(os.path.join(args.dicom_dir, patient))

    # Preprocess nifti files
    for root, dirs, files in os.walk(args.nifti_dir):
        for file in files:
            try:
                preprocess_nifti(os.path.join(root, file), os.path.join(root, 'processed.nii.gz'))
            except:
                print("Error occurred for file: {}".format(file))

    # Delete unprocessed nifti files
    if args.delete_unprocessed:
        for root, dirs, files in os.walk(args.nifti_dir):
            for file in files:
                if not file == 'processed.nii.gz':
                    os.remove(os.path.join(root, file))
