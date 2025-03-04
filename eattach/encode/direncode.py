import logging
import mimetypes
import os.path
from typing import Callable, Any, Union, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

from .encoder import get_encoder

mimetypes.add_type('image/webp', '.webp')


class UnsupportedFileFormat(Exception):
    pass


def is_npy_file(file: str):
    _, ext = os.path.splitext(file.lower())
    return ext == '.npy'


def is_npz_file(file: str):
    _, ext = os.path.splitext(file.lower())
    return ext == '.npz'


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


def get_npz_file(filename: str):
    return os.path.splitext(filename)[0] + '.npz'


def npz_file_append(npz_file: str, extra_data: Dict[str, Any]):
    data = {}
    if os.path.exists(npz_file):
        with np.load(npz_file) as nf:
            data.update(nf)

    data.update(**extra_data)
    np.savez(npz_file, extra_data)


def npz_file_key_exists(npz_file: str, key: str) -> bool:
    if os.path.exists(npz_file):
        with np.load(npz_file) as nf:
            return key in nf
    else:
        return False


def get_embedding_key(encode_system: str, model_name: str) -> str:
    return f'{encode_system}:{model_name}'


def encode_for_dir(directory: str, encode_system: str, model_name: str,
                   modality: Union[str, Callable[[str], Any]] = 'image'):
    encoder = get_encoder(encode_system=encode_system, model_name=model_name)
    encode_system, model_name = encoder.encode_system, encoder.model_name
    modality = _LOADERS.get(modality, modality)
    embedding_key = get_embedding_key(encode_system=encode_system, model_name=model_name)

    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.abspath(os.path.join(root, file))
            if is_npz_file(path):
                continue

            dst_npz_path = get_npz_file(path)
            if npz_file_key_exists(dst_npz_path, embedding_key):
                continue

            paths.append(path)

    for path in tqdm(paths, desc=f'Encode {directory!r} with {encode_system!r}'):
        dst_npz_path = get_npz_file(path)
        if npz_file_key_exists(dst_npz_path, embedding_key):
            continue

        try:
            data = modality(path)
        except UnsupportedFileFormat:
            logging.warning(f'Unsupported file encoding - {path}.')
            continue

        binary_data = encoder(data)
        npz_file_append(dst_npz_path, {embedding_key: binary_data})
