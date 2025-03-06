import glob
import mimetypes
import os.path
from typing import List, Optional

from imgutils.data import load_image
from torch.utils.data import Dataset


class ImageDirDataset(Dataset):
    def __init__(self, image_dir, labels: List[str], unsupervised: Optional[str] = None, transform=None,
                 no_cache: bool = False):
        self.labels = labels
        self._label_map = {l: i for i, l in enumerate(labels)}

        self.images = []
        for lid, label in enumerate(labels):
            for f in glob.glob(os.path.join(image_dir, label, '**', '*'), recursive=True):
                mimetype, _ = mimetypes.guess_type(f)
                if mimetype and mimetype.startswith('image/'):
                    self.images.append((f, lid))
        if unsupervised:
            for f in glob.glob(os.path.join(image_dir, unsupervised, '**', '*'), recursive=True):
                mimetype, _ = mimetypes.guess_type(f)
                if mimetype and mimetype.startswith('image/'):
                    self.images.append((f, -1))

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
