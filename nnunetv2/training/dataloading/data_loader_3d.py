import os.path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
import SimpleITK as sitk
from nnunetv2.paths import nnUNet_results

def tst_output(pid, tmp, tag):
    output = torch.cat((tmp['image'], tmp['regression_target'], tmp['segmentation'][0]), dim=0)
    for i in range(output.shape[0]):
        # Convert the numpy array to a SimpleITK image
        sitk_image = sitk.GetImageFromArray(output.numpy()[i])

        # Optionally set the spacing and origin (if needed)
        sitk_image.SetSpacing((1.0, 1.0, 1.0))  # Set spacing for x, y, z dimensions
        sitk_image.SetOrigin((0.0, 0.0, 0.0))  # Set origin for x, y, z dimensions

        unique_values_label = np.unique(tmp['segmentation'][0])
        if np.array_equal(unique_values_label, np.array([0., 1.])):
            label_range = 'label01'
        else:
            label_range = 'label0'
        # Save the image as a .nii.gz file
        sitk.WriteImage(sitk_image, os.path.join(nnUNet_results, fr'{tag}_{pid}_{i}_{label_range}.nii.gz'))


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
                        unique_values = np.unique(data_all[b][-1:])
                        if not np.array_equal(unique_values, np.array([0., 1., 2.])):
                            print(f'pid={selected_keys[b]}, before self.transforms, zonal_mask np.unique is {unique_values}')
                        # tmp solution: only applicable when the last input channel is mask
                        tmp = self.transforms(**{'image': data_all[b][:-1], 'regression_target': data_all[b][-1:], 'segmentation': seg_all[b]})
                        unique_values = np.unique(tmp['regression_target'][0])
                        if not np.array_equal(unique_values, np.array([0., 1., 2.])):
                            print(f"pid={selected_keys[b]}, after self.transforms, zonal_mask np.unique is {unique_values}")
                            tst_output(selected_keys[b], tmp, 'badcase')
                        # else:
                        #     tst_output(selected_keys[b], tmp, 'goodcase')
                        combined_tensor = torch.cat((tmp['image'], tmp['regression_target']), dim=0)
                        images.append(combined_tensor)
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
