import glob
import os.path

import numpy as np
from torch.utils.data import Dataset


class InstanceDirDataset(Dataset):
    def __init__(self, obj_dir, labels, embedding_key: str, no_cache: bool = False):
        self.labels = labels
        self._label_map = {l: i for i, l in enumerate(labels)}
        self._embedding_key = embedding_key

        self.objs = []
        for lid, label in enumerate(labels):
            for f in glob.glob(os.path.join(obj_dir, label, '*.npz')):
                self.objs.append((f, lid))

        self._cached_objs = {}
        self._no_cache = no_cache

    def _raw_load_obj(self, index):
        obj_file, lid = self.objs[index]
        with np.load(obj_file) as nf:
            obj = nf[self._embedding_key]
        return obj, lid

    def _getitem(self, index):
        if self._no_cache:
            obj, lid = self._raw_load_obj(index)
        else:
            if index not in self._cached_objs:
                obj, lid = self._raw_load_obj(index)
                self._cached_objs[index] = (obj, lid)
            else:
                obj, lid = self._cached_objs[index]

        return obj, lid

    def __getitem__(self, index):
        obj, lid = self._getitem(index)
        return obj, lid

    def __len__(self):
        return len(self.objs)
