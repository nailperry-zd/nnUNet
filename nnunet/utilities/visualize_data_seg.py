import os.path
import SimpleITK as sitk
from nnunet.paths import *
FLAG_BEFORE_AUG = False
FLAG_AFTER_AUG = False
FLAG_DATA_FOR_PREDICTION = False
import time

def save_volume_default(volume_arr, target_output):
    # Convert to SimpleITK image
    sitk_image = sitk.GetImageFromArray(volume_arr)

    # Set the direction and origin if needed (identity in this case, but can be customized)
    sitk_image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])  # Identity matrix for direction
    sitk_image.SetOrigin([0.0, 0.0, 0.0])  # Default origin

    # Save the image
    sitk.WriteImage(sitk_image, target_output)

def visualize_data_seg(data, seg, selected_keys):
    if not FLAG_BEFORE_AUG:
        return
    for b in range(data.shape[0]):
        output_folder = os.path.join(network_training_output_dir, 'data_before_augmentation')
        os.makedirs(output_folder, exist_ok=True)
        # Loop over the volumes and save them
        for i in range(seg.shape[1]):
            # Process each volume (18, 124, 124)
            volume = seg[b][i]
            save_volume_default(volume, os.path.join(output_folder, fr'{selected_keys[b]}_seg_{i + 1}.nii.gz'))

        # Loop over the volumes and save them
        for i in range(data.shape[1]):
            # Process each volume (18, 124, 124)
            volume = data[b][i]
            save_volume_default(volume, os.path.join(output_folder, fr'{selected_keys[b]}_{i + 1}.nii.gz'))

def visualize_data_for_prediction(data, output_file_path):
    if not FLAG_DATA_FOR_PREDICTION:
        return
    # Extract the directory and filename
    directory = os.path.dirname(output_file_path)
    filename = os.path.basename(output_file_path)
    os.makedirs(directory, exist_ok=True)
    # Retrieve the base filename without extension
    base_filename = os.path.splitext(filename)[0]  # e.g., 'rstrial_058'
    for b in range(data.shape[0]):
        # Construct the new file path
        new_file_path = os.path.join(directory, f"{base_filename}_{b}.nii.gz")
        # Process each volume (18, 124, 124)
        volume = data[b]
        save_volume_default(volume, new_file_path)


def visualize_data_seg_aug(data_list, seg_list):
    if not FLAG_AFTER_AUG:
        return
    output_folder = os.path.join(network_training_output_dir, 'data_after_augmentation')
    os.makedirs(output_folder, exist_ok=True)
    timestamp = time.time()
    # the dtype of each element is Tensor
    for index, tensor in enumerate(data_list):
        print(f'The size of data_ is {tensor.size()}')
        # Loop over the volumes and save them
        for i in range(tensor.size(0)):
            # Process each volume (18, 124, 124)
            slice_tensor = tensor[i]
            # Convert the PyTorch tensor to a NumPy array
            volume = slice_tensor.numpy()
            save_volume_default(volume, os.path.join(output_folder, fr'{timestamp}_b{index}_{i + 1}.nii.gz'))

    seg_tensor_origin_size = seg_list[0]# torch.Size([2, 1, 16, 320, 320])

    for index in range(seg_tensor_origin_size.size(0)):
        seg_curr = seg_tensor_origin_size[index]
        slice_tensor = seg_curr.squeeze(0)  # Remove dimension at index 0
        # Convert the PyTorch tensor to a NumPy array
        volume = slice_tensor.numpy()
        save_volume_default(volume, os.path.join(output_folder, fr'{timestamp}_b{index}_seg.nii.gz'))
