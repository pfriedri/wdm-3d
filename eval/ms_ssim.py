import argparse
import numpy as np
import torch
import sys

sys.path.append(".")
sys.path.append("..")

from generative.metrics import MultiScaleSSIMMetric
from monai import transforms
from monai.config import print_config
from monai.data import Dataset
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from tqdm import tqdm
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.lidcloader import LIDCVolumes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    parser.add_argument("--sample_dir", type=str, required=True, help="Location of the samples to evaluate.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--dataset", choices=['brats','lidc-idri'], required=True, help="Dataset (brats | lidc-idri)")
    parser.add_argument("--img_size", type=int, required=True)

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    #print_config()

    if args.dataset == 'brats':
        dataset_1 = BRATSVolumes(directory=args.sample_dir, mode='fake', img_size=args.img_size)
        dataset_2 = BRATSVolumes(directory=args.sample_dir, mode='fake', img_size=args.img_size)

    elif args.dataset == 'lidc-idri':
        dataset_1 = LIDCVolumes(directory=args.sample_dir, mode='fake', img_size=args.img_size)
        dataset_2 = LIDCVolumes(directory=args.sample_dir, mode='fake', img_size=args.img_size)


    dataloader_1 = DataLoader(dataset_1, batch_size=1, shuffle=False, num_workers=args.num_workers)
    dataloader_2 = DataLoader(dataset_2, batch_size=1, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda")
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=7)

    print("Computing MS-SSIM (this takes a while)...")
    ms_ssim_list = []
    pbar = tqdm(enumerate(dataloader_1), total=len(dataloader_1))
    for step, batch in pbar:
        img = batch[0]
        for batch2 in dataloader_2:
            img2 = batch2 [0]
            if batch[1] == batch2[1]:
                continue
            ms_ssim_list.append(ms_ssim(img.to(device), img2.to(device)).item())
        pbar.update()

    ms_ssim_list = np.array(ms_ssim_list)
    print("Calculated MS-SSIMs. Computing mean ...")
    print(f"Mean MS-SSIM: {ms_ssim_list.mean():.6f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
