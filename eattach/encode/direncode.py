import logging
import mimetypes
import os.path
from ctypes import Union
from typing import Callable, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from .encoder import get_encoder

mimetypes.add_type('image/webp', '.webp')


class UnsupportedFileFormat(Exception):
    pass


def is_npy_file(file: str):
    _, ext = os.path.splitext(file)
    return ext in {'.npy', '.npz'}


def _fn_load_image(file: str):
    mimetype, _ = mimetypes.guess_type(file)
    if mimetype and mimetype.startswith('image/'):
        image = Image.open(file)
        image.load()
        return image
    else:
        raise UnsupportedFileFormat(f'Unsupported image file - {file!r}.')


_LOADERS = {
    'image': _fn_load_image,
}


def encode_for_dir(directory: str, encode_system: str, model_name: str,
                   modality: Union[str, Callable[[str], Any]] = 'image'):
    encoder = get_encoder(encode_system=encode_system, model_name=model_name)
    modality = _LOADERS.get(modality, modality)

    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.abspath(os.path.join(root, file))
            if is_npy_file(path):
                continue

            dst_npy_path = os.path.splitext(path)[0] + '.npy'
            if os.path.exists(dst_npy_path):
                continue

            paths.append(path)

    for path in tqdm(paths, desc=f'Encode {directory!r} with {encode_system!r}'):
        dst_npy_path = os.path.splitext(path)[0] + '.npy'
        if os.path.exists(dst_npy_path):
            continue

        try:
            data = modality(path)
        except UnsupportedFileFormat:
            logging.warning(f'Supported file encoding - {path}.')
            continue

        binary_data = encoder(data)
        np.save(dst_npy_path, binary_data)
