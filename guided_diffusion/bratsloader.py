import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel


class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False, normalize=None, mode='train', img_size=256):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_NNN_XXX_123_w.nii.gz
                  where XXX is one of t1n, t1c, t2w, t2f, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.test_flag = test_flag
        self.img_size = img_size
        if test_flag:
            self.seqtypes = ['t1n', 't1c', 't2w', 't2f']
        else:
            self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        self.seqtypes_set = set(self.seqtypes)
        self.database = []

        if not self.mode == 'fake': # Used during training and for evaluating real data
            for root, dirs, files in os.walk(self.directory):
                # if there are no subdirs, we have a datadir
                if not dirs:
                    files.sort()
                    datapoint = dict()
                    # extract all files as channels
                    for f in files:
                        seqtype = f.split('-')[4].split('.')[0]
                        datapoint[seqtype] = os.path.join(root, f)
                    self.database.append(datapoint)
        else:   # Used for evaluating fake data
            for root, dirs, files in os.walk(self.directory):
                for f in files:
                    datapoint = dict()
                    datapoint['t1n'] = os.path.join(root, f)
                    self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        name = filedict['t1n']
        nib_img = nibabel.load(name)  # We only use t1 weighted images
        out = nib_img.get_fdata()

        if not self.mode == 'fake':
            # CLip and normalize the images
            out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
            out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
            out = torch.tensor(out_normalized)

            # Zero pad images
            image = torch.zeros(1, 256, 256, 256)
            image[:, 8:-8, 8:-8, 50:-51] = out

            # Downsampling
            if self.img_size == 128:
                downsample = nn.AvgPool3d(kernel_size=2, stride=2)
                image = downsample(image)
        else:
            image = torch.tensor(out, dtype=torch.float32)
            image = image.unsqueeze(dim=0)

        # Normalization
        image = self.normalize(image)

        if self.mode == 'fake':
            return image, name
        else:
            return image

    def __len__(self):
        return len(self.database)
