import csv
import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Type, Union, Tuple

import PIL
from sklearn import datasets
import torch
import numpy as np
import pandas as pd

from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg, extract_archive
from torchvision.datasets.vision import VisionDataset

from PIL import Image


def add_feature(img, aug='uf', imshow=False, resize=None):
    # float version
    uf = [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
    ]

    size = 5

    uf = np.float32(uf)# * 255
    if resize is not None:
        uf = Image.fromarray(uf)
        uf = uf.resize(resize, resample=PIL.Image.Resampling.NEAREST)
        uf = np.asarray(uf, dtype=np.float32)
        size = resize[0]

    uf = torch.FloatTensor(uf)
    uf = torch.stack([uf] * 3, axis=0)

    offset = 4

    if aug == 'random':
        f = torch.rand(size=(3, size, size), dtype=torch.float32)
    elif aug == 'invert':
        f = torch.where(uf==1, 0, 1)
    elif aug == 'uf':
        f = uf
    elif aug == 'clean':
        return img
    else:
        raise NotImplementedError(aug)

    #img = np.asarray(img)
    #f = np.asarray(f)
    img[:, offset:size + offset, offset:size + offset] = f
    #img = PIL.Image.fromarray(img)

    if imshow:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(torch.permute(img, dims=(1, 2, 0)))
        plt.savefig('debug_celeba/ufimg.png')
        plt.close()

    return img

class CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        canary_id = None,
        label_type = 'attractive',
        n = None,
        resize=None,
        shuffle_labels = False,
        n_classes=2
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.canary_id = canary_id
        self.label_type = label_type
        self.resize = resize
        self.shuffle_labels = shuffle_labels
        self.n_classes = n_classes
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        #if download:
        #    self.download()

        #if not self._check_integrity():
        #    raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        if label_type == 'attractive':
            data_sheet = 'list_attr_attractive_celeba.csv'
        else:
            data_sheet = 'list_attr_haircolor_celeba.csv'
        df = pd.read_csv(os.path.join(self.root, self.base_folder, data_sheet))
        df = df.loc[df['split'] == split].reindex()
        self.df = df

        if self.shuffle_labels:
            self.df['label'] = np.random.randint(0, self.n_classes, len(self.df))

        if n is not None:
            self.df = self.df[:n]

        self.feature = 'uf'

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        extract_archive(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        row =  self.df.iloc[index]
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", row['Filename']))

        if self.transform is not None:
            X = self.transform(X)
        
        if type(self.canary_id) == str:
            if self.canary_id == 'all':
                X = add_feature(X, aug=self.feature, resize=self.resize)
        elif type(self.canary_id) == np.ndarray:
            if self.canary_id is not None and index in self.canary_id:
                X = add_feature(X, aug=self.feature, resize=self.resize)
        elif self.canary_id is None:
            pass
        else:
            raise Exception(f'Unknown canary id type {type(self.canary_id)}')

        target = row['label']

        return X, target, index

    def __len__(self) -> int:
        return len(self.df)
    
    def get_targets(self):
        labels = self.df['label'].to_numpy()
        return labels