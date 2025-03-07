import glob
import mimetypes
import os.path
from dataclasses import dataclass
from typing import List, Optional

from imgutils.data import load_image
from natsort import natsorted
from torch.utils.data import Dataset


@dataclass
class ImageDirLabels:
    labels: List[str]
    unsupervised: Optional[str] = None


def load_labels_from_image_dir(image_dir: str, unsupervised: Optional[str] = '__unlabeled__') -> ImageDirLabels:
    labels: List[str] = []
    uns_name: Optional[str] = None
    for d in natsorted(os.listdir(image_dir)):
        dir_ = os.path.join(image_dir, d)
        if os.path.isdir(dir_):
            has_image = False
            for f in glob.glob(os.path.join(dir_, '**', '*'), recursive=True):
                mimetype, _ = mimetypes.guess_type(f)
                if mimetype and mimetype.startswith('image/'):
                    has_image = True
                    break

            if has_image:
                if unsupervised and d == unsupervised:
                    uns_name = d
                else:
                    labels.append(d)

    return ImageDirLabels(
        labels=labels,
        unsupervised=uns_name,
    )


class ImageDirDataset(Dataset):
    def __init__(self, image_dir: str, labels: List[str], transform=None,
                 no_cache: bool = False):
        self.labels = labels
        self._label_map = {l: i for i, l in enumerate(labels)}

        self.images = []
        for lid, label in enumerate(labels):
            for f in glob.glob(os.path.join(image_dir, label, '**', '*'), recursive=True):
                mimetype, _ = mimetypes.guess_type(f)
                if mimetype and mimetype.startswith('image/'):
                    self.images.append((f, lid))

        self.transform = transform

        self._cached_images = {}
        self._no_cache = no_cache

    def _raw_load_image(self, index):
        image_file, lid = self.images[index]
        image = load_image(image_file, force_background='white', mode='RGB')
        return image, lid

    def _getitem(self, index):
        if self._no_cache:
            image, lid = self._raw_load_image(index)
        else:
            if index not in self._cached_images:
                image, lid = self._raw_load_image(index)
                self._cached_images[index] = (image, lid)
            else:
                image, lid = self._cached_images[index]

        return image, lid

    def __getitem__(self, index):
        image, lid = self._getitem(index)
        if self.transform is not None:
            image = self.transform(image)
        return image, lid

    def __len__(self):
        return len(self.images)


class ImageDirUnsupervisedDataset(Dataset):
    def __init__(self, image_dir: str, weak_transform=None, strong_transform=None,
                 no_cache: bool = False):
        self.images = []
        for f in glob.glob(os.path.join(image_dir, '**', '*'), recursive=True):
            mimetype, _ = mimetypes.guess_type(f)
            if mimetype and mimetype.startswith('image/'):
                self.images.append(f)

        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        self._cached_images = {}
        self._no_cache = no_cache

    def _raw_load_image(self, index):
        image_file = self.images[index]
        image = load_image(image_file, force_background='white', mode='RGB')
        return image

    def _getitem(self, index):
        if self._no_cache:
            image = self._raw_load_image(index)
        else:
            if index not in self._cached_images:
                image = self._raw_load_image(index)
                self._cached_images[index] = image
            else:
                image = self._cached_images[index]

        return image

    def __getitem__(self, index):
        image = self._getitem(index)
        weak_image, strong_image = image, image
        if self.weak_transform is not None:
            weak_image = self.weak_transform(weak_image)
        if self.strong_transform is not None:
            strong_image = self.strong_transform(strong_image)
        return weak_image, strong_image

    def __len__(self):
        return len(self.images)
