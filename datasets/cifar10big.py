import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
import PIL
from PIL import Image

import torch
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

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

    return img

class CIFAR10Big(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        canary_id = None,
        n_classes = 2,
        shuffle_labels = True,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        self.canary_id = canary_id
        self.feature = None
        self.resize = None

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

        # random target labels
        if shuffle_labels:
            self.targets = np.random.choice(np.arange(n_classes), size=self.data.shape[0])
        else:
            self.targets = np.int64(self.targets)

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if type(self.canary_id) == str:
            if self.canary_id == 'all':
                img = add_feature(img, aug=self.feature, resize=self.resize)
        elif type(self.canary_id) == np.ndarray:
            if self.canary_id is not None and index in self.canary_id:
                img = add_feature(img, aug=self.feature, resize=self.resize)
        elif self.canary_id is None:
            pass
        else:
            raise Exception(f'Unknown canary id type {type(self.canary_id)}')

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    def get_targets(self):
        labels = self.targets
        return labels

