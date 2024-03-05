import torch
import torch.nn as nn
import torch.utils.data
import os
import os.path
import nibabel


class LIDCVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False, normalize=None, mode='train', img_size=256):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have the name: processed.nii.gz
        '''
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.test_flag = test_flag
        self.img_size = img_size
        self.database = []

        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have a datadir
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    datapoint['image'] = os.path.join(root, f)
                if len(datapoint) != 0:
                    self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        nib_img = nibabel.load(filedict['image'])
        out = nib_img.get_fdata()
        out = torch.Tensor(out)

        image = torch.zeros(1, 256, 256, 256)
        image[:, :, :, :] = out

        if self.img_size == 128:
            downsample = nn.AvgPool3d(kernel_size=2, stride=2)
            image = downsample(image)

        # normalization
        image = self.normalize(image)

        return image

    def __len__(self):
        return len(self.database)
