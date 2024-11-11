import os.path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
import SimpleITK as sitk
from nnunetv2.paths import nnUNet_results

DEBUG_FLAG = False
CUSTOM_OUTPUT_PATH = r"C:\Users\dzha937\DEV\pycharm_workdir\Training_in_Progress\Dataset912_ProstatexCT"
# CUSTOM_OUTPUT_PATH = None
def tst_output(pid, output, label, tag):
    if not DEBUG_FLAG:
        return
    unique_values_label = np.unique(label)
    if np.array_equal(unique_values_label, np.array([0., 1.])):
        label_range = 'label01'
    else:
        label_range = 'label0'
    for i in range(output.shape[0]):
        # Convert the numpy array to a SimpleITK image
        sitk_image = sitk.GetImageFromArray(output.numpy()[i])

        # Optionally set the spacing and origin (if needed)
        sitk_image.SetSpacing((1.0, 1.0, 1.0))  # Set spacing for x, y, z dimensions
        sitk_image.SetOrigin((0.0, 0.0, 0.0))  # Set origin for x, y, z dimensions

        # Save the image as a .nii.gz file
        output_dir = nnUNet_results if CUSTOM_OUTPUT_PATH is None else CUSTOM_OUTPUT_PATH
        sitk.WriteImage(sitk_image, os.path.join(output_dir, fr'{tag}_{pid}_{i}_{label_range}.nii.gz'))

    sitk_image = sitk.GetImageFromArray(label)

    # Optionally set the spacing and origin (if needed)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))  # Set spacing for x, y, z dimensions
    sitk_image.SetOrigin((0.0, 0.0, 0.0))  # Set origin for x, y, z dimensions
    output_dir = nnUNet_results if CUSTOM_OUTPUT_PATH is None else CUSTOM_OUTPUT_PATH
    sitk.WriteImage(sitk_image, os.path.join(output_dir, fr'{tag}_{pid}_label_{label_range}.nii.gz'))

def is_c0003_mask(data):
    channel_num = data.shape[0]
    # unique_values = np.unique(data[-1:])
    return channel_num > 3

class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        if is_c0003_mask(data_all[b]):
                            # print(f'pid={selected_keys[b]}, before self.transforms, zonal_mask np.unique is {unique_values}')
                            # tmp solution: only applicable when the last input channel is mask
                            # stack [label, mask] together
                            tmp_mask = torch.cat((seg_all[b], data_all[b][-1:]), dim=0)
                            tmp = self.transforms(**{'image': data_all[b][:-1], 'segmentation': tmp_mask})
                            processed_mask = tmp['segmentation'][0][-1:]
                            combined_tensor = torch.cat((tmp['image'], processed_mask), dim=0)
                            tst_output(selected_keys[b], combined_tensor, tmp['segmentation'][0][0], 'treat_last_channel_as_mask')
                            images.append(combined_tensor)

                            processed_label_pyramid = []
                            for i in range(len(tmp['segmentation'])):
                                processed_label_pyramid.append(tmp['segmentation'][i][0:1])

                            segs.append(processed_label_pyramid)
                        else:
                            tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                            tst_output(selected_keys[b], tmp['image'], tmp['segmentation'][0], 'conventional')
                            images.append(tmp['image'])
                            segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

            return {'data': data_all, 'target': seg_all, 'keys': selected_keys}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
