import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
import argparse

sys.path.append(".")
sys.path.append("..")

from scipy import linalg

from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.lidcloader import LIDCVolumes
from model import generate_model


def get_feature_extractor(sets):
    model, _ = generate_model(sets)
    checkpoint = torch.load(sets.pretrain_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Done. Initialized feature extraction model and loaded pretrained weights.")

    return model


def get_activations(model, data_loader, sets):
    pred_arr = np.empty((sets.num_samples, sets.dims))

    for i, batch in enumerate(data_loader):
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.cuda()
        if i % 10 == 0:
            print('\rPropagating batch %d' % i, end='', flush=True)
        with torch.no_grad():
            pred = model(batch)

        if i*sets.batch_size >= pred_arr.shape[0]:
            pred_arr[i*sets.batch_size:] = pred.cpu().numpy()
            break
        else:
            pred_arr[i*sets.batch_size:(i+1)*sets.batch_size] = pred.cpu().numpy()

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance. The Frechet distance between two multivariate Gaussians
    X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the inception net (like returned by the function
               'get_predictions') for generated samples.
    -- mu2   : The sample mean over activations, precalculated on a representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on a representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def process_feature_vecs(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='Dataset (brats | lidc-idri)')
    parser.add_argument('--img_size', required=True, type=int, help='Image size')
    parser.add_argument('--data_root_real', required=True, type=str, help='Path to real data')
    parser.add_argument('--data_root_fake', required=True, type=str, help='Path to fake data')
    parser.add_argument('--pretrain_path', required=True, type=str, help='Path to pretrained model')
    parser.add_argument('--path_to_activations', required=True, type=str, help='Path to activations')
    parser.add_argument('--n_seg_classes', default=2, type=int, help="Number of segmentation classes")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of jobs')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('--phase', default='test', type=str, help='Phase of train or test')
    parser.add_argument('--save_intervals', default=10, type=int, help='Interation for saving model')
    parser.add_argument('--n_epochs', default=200, type=int, help='Number of total epochs to run')
    parser.add_argument('--input_D', default=256, type=int, help='Input size of depth')
    parser.add_argument('--input_H', default=256, type=int, help='Input size of height')
    parser.add_argument('--input_W', default=256, type=int, help='Input size of width')
    parser.add_argument('--resume_path', default='', type=str, help='Path for resume model.')

    parser.add_argument('--new_layer_names', default=['conv_seg'], type=list, help='New layer except for backbone')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--gpu_id', default=0, type=int, help='Gpu id')
    parser.add_argument('--model', default='resnet', type=str,
                        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=50, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--ci_test', action='store_true', help='If true, ci testing is used.')
    args = parser.parse_args()
    args.save_folder = "./trails/models/{}_{}".format(args.model, args.model_depth)

    return args


if __name__ == '__main__':
    # Model settings
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'
    sets.batch_size = 1
    sets.dims = 2048
    sets.num_samples = 1000

    if not sets.no_cuda:
        dev_name = 'cuda:' + str(sets.gpu_id)
        device = torch.device(dev_name)
    else:
        device = torch.device('cpu')

    # getting model
    print("Load model ...")
    model = get_feature_extractor(sets)
    model = model.to(device)

    # Data loader
    print("Initialize dataloader ...")
    if sets.dataset == 'brats':
        real_data = BRATSVolumes(sets.data_root_real, normalize=None, mode='real', img_size=sets.img_size)
        fake_data = BRATSVolumes(sets.data_root_fake, normalize=None, mode='fake', img_size=sets.img_size)

    elif sets.dataset == 'lidc-idri':
        real_data = LIDCVolumes(sets.data_root_real, normalize=None, mode='real', img_size=sets.img_size)
        fake_data = LIDCVolumes(sets.data_root_fake, normalize=None, mode='fake', img_size=sets.img_size)

    else:
        print("Dataloader for this dataset is not implemented. Use 'brats' or 'lidc-idri'.")

    real_data_loader = DataLoader(real_data, batch_size=sets.batch_size, shuffle=False, num_workers=sets.batch_size,
                                  pin_memory=False)
    fake_data_loader = DataLoader(fake_data, batch_size=sets.batch_size, shuffle=False, num_workers=sets.batch_size,
                                  pin_memory=False)


    # Real data
    print("Get activations from real data ...")
    activations_real = get_activations(model, real_data_loader, sets)
    mu_real, sigma_real = process_feature_vecs(activations_real)

    path_to_mu_real = os.path.join(sets.path_to_activations, 'mu_real.npy')
    path_to_sigma_real = os.path.join(sets.path_to_activations, 'sigma_real.npy')
    np.save(path_to_mu_real, mu_real)
    print("")
    print("Saved mu_real to: " + path_to_mu_real)
    np.save(path_to_sigma_real, sigma_real)
    print("Saved sigma_real to: " + path_to_sigma_real)


    # Fake data
    print("Get activations from fake/generated data ...")
    activations_fake = get_activations(model, fake_data_loader, sets)
    mu_fake, sigma_fake = process_feature_vecs(activations_fake)

    path_to_mu_fake = os.path.join(sets.path_to_activations, 'mu_fake.npy')
    path_to_sigma_fake = os.path.join(sets.path_to_activations, 'sigma_fake.npy')
    np.save(path_to_mu_fake, mu_fake)
    print("")
    print("Saved mu_fake to: " + path_to_mu_fake)
    np.save(path_to_sigma_fake, sigma_fake)
    print("Saved sigma_fake to: " + path_to_sigma_fake)

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    print("The FID score is: ")
    print(fid)
