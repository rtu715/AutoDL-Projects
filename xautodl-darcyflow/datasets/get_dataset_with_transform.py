##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os
import sys
import torch
import os.path as osp
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import boto3
from copy import deepcopy
from PIL import Image

from xautodl.config_utils import load_config

from .DownsampledImageNet import ImageNet16
from .SearchDatasetWrap import SearchDataset
from .download_data import download_from_s3
from .utils_pde import LpLoss, MatReader, UnitGaussianNormalizer, create_grid

Dataset2Class = {
    "darcyflow": 1,
}
s3_bucket = "pde-xd"


def get_datasets(name, root, cutout):

    # Data Argumentation
    if name == 'darcyflow':
        xshape = (1, 85, 85, 3)
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == "darcyflow":
        s3 = boto3.client("s3")
        path = os.path.join(root, 'darcyflow_data')
        os.makedirs(path, exist_ok=True)
        download_from_s3(s3_bucket, name, path)
        TRAIN_PATH = os.path.join(path, 'piececonst_r421_N1024_smooth1.mat')
        reader = MatReader(TRAIN_PATH)
        r = 5
        grid, s = create_grid(r)
        ntrain = 1000
        ntest = 100
        x_train = reader.read_field('coeff')[:ntrain-ntest, ::r, ::r][:, :s, :s]
        y_train = reader.read_field('sol')[:ntrain-ntest, ::r, ::r][:, :s, :s]

        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)

        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

        x_train = torch.cat(
            [x_train.reshape(ntrain-ntest, s, s, 1), grid.repeat(ntrain-ntest, 1, 1, 1)], dim=3)
        train_data = torch.utils.data.TensorDataset(x_train, y_train)

        x_valid = reader.read_field('coeff')[ntrain-ntest:ntrain, ::r, ::r][:, :s, :s]
        y_valid = reader.read_field('sol')[ntrain-ntest:ntrain, ::r, ::r][:, :s, :s]
        x_valid = x_normalizer.encode(x_valid)
        x_valid = torch.cat([x_valid.reshape(ntest, s, s, 1), grid.repeat(ntest, 1, 1, 1)], dim=3)
        valid_data = torch.utils.data.TensorDataset(x_valid, y_valid)

        TEST_PATH = os.path.join(
            path, 'piececonst_r421_N1024_smooth2.mat')
        reader = MatReader(TEST_PATH)
        x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
        y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

        x_test = x_normalizer.encode(x_test)
        x_test = torch.cat([x_test.reshape(ntest, s, s, 1),
                           grid.repeat(ntest, 1, 1, 1)], dim=3)
        test_data = torch.utils.data.TensorDataset(x_test, y_test)

    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    normalizer = y_normalizer if name == 'darcyflow' else None
    class_num = Dataset2Class[name]

    return train_data, valid_data, test_data, xshape, class_num, normalizer


def get_nas_search_loaders(
    train_data, valid_data, dataset, config_root, batch_size, workers
):
    if isinstance(batch_size, (list, tuple)):
        batch, test_batch = batch_size
    else:
        batch, test_batch = batch_size, batch_size
    if dataset == "darcyflow":
        search_train_data = train_data
        search_valid_data = valid_data
        search_data = SearchDataset(
            dataset,
            [search_train_data, search_valid_data],
            list(range(900)),
            list(range(100)),
        )
        search_loader = torch.utils.data.DataLoader(
            search_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=test_batch,
            num_workers=workers,
            pin_memory=True,
        )
    else:
        raise ValueError("invalid dataset : {:}".format(dataset))
    return search_loader, train_loader, valid_loader


# if __name__ == '__main__':
#  train_data, test_data, xshape, class_num = dataset = get_datasets('cifar10', '/data02/dongxuanyi/.torch/cifar.python/', -1)
#  import pdb; pdb.set_trace()
