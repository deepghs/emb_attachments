import json
import os
from typing import Optional, Callable

from accelerate import Accelerator
from ditk import logging
from hbutils.random import global_seed

from ...dataset import ImageDirDataset, dataset_split, WrappedImageDataset, load_labels_from_image_dir
from ...encode import load_encoder, EncoderModel
from ...model import Backbone, TrainModel
from ...problem import ClassificationProblem


def train_classification(
        workdir: str,
        dataset_dir: str,

        # model configuration
        encoder_model: str = 'wdtagger:SmilingWolf/wd-swinv2-tagger-v3',
        model_type: str = 'mlp',
        init_params: dict = None,

        # dataset configuration
        train_augment: Optional[Callable] = None,
        test_split_ratio: float = 0.2,

        # train hyperparams
        max_epoch: int = 100,
        batch_size: int = 16,
        seed: Optional[int] = 0,
):
    if seed is not None:
        # native random, numpy, torch and faker's seeds are includes
        # if you need to register more library for seeding, see:
        # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
        logging.info(f'Globally set the random seed {seed!r}.')
        global_seed(seed)

    os.makedirs(workdir, exist_ok=True)

    init_params = dict(init_params or {})
    checkpoints = os.path.join(workdir, 'checkpoints')
    last_ckpt_zip_file = os.path.join(checkpoints, 'last.zip')
    if os.path.exists(last_ckpt_zip_file):
        logging.info(f'Loading last checkpoint from {last_ckpt_zip_file!r} ...')
        backbone, metadata = Backbone.load_from_zip(last_ckpt_zip_file, with_metadata=True)
        if model_type != backbone.type:
            raise RuntimeError('Model type not match with the previous version, '
                               f'{backbone.type!r} expected but {model_type!r} found.')
        if init_params != backbone.init_params:
            raise RuntimeError('Init params not match, '
                               f'{backbone.init_params!r} expected but {init_params!r} found.')
        last_epoch = metadata['train/epoch']
        logging.info(f'Resume from epoch {last_epoch!r}.')
    else:
        logging.info(f'No last checkpoint found, initialize {model_type!r} model with params {init_params!r}.')
        backbone = Backbone.new(type_=model_type, **init_params)
        last_epoch = 0
    model_type, init_params = backbone.type, backbone.init_params

    encoder: EncoderModel = load_encoder(encoder_model)

    train_cfg = {
        'batch_size': batch_size,
        'max_epoch': max_epoch,
        'seed': seed,
    }
    with open(os.path.join(workdir, 'meta.json'), 'w') as f:
        json.dump({
            'problem': 'classification',
            'model_type': model_type,
            'init_params': init_params,
            'encoder_model': encoder_model,
        }, f, indent=4, ensure_ascii=False, sort_keys=True)

    labels_info = load_labels_from_image_dir(dataset_dir)
    logging.info(f'Load labels from dataset directory {dataset_dir!r}, '
                 f'labels: {labels_info.labels!r}, unsupervised: {labels_info.unsupervised!r}.')
    problem = ClassificationProblem(labels=labels_info.labels)
    dataset = ImageDirDataset(dataset_dir, labels=labels_info.labels, unsupervised=labels_info.unsupervised,
                              no_cache=True)
    train_dataset, test_dataset = dataset_split(dataset, [1 - test_split_ratio, test_split_ratio])
    if train_augment:
        train_dataset = WrappedImageDataset(train_dataset, train_augment)
    train_dataset = WrappedImageDataset(train_dataset, encoder.preprocessor)
    test_dataset = WrappedImageDataset(test_dataset, encoder.preprocessor)

    model = TrainModel(
        encoder=encoder.model,
        backbone=backbone.module,
        head=None,
    )
    print(model)

    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    train_classification(
        workdir='runs/train_test',
        dataset_dir='/data/monochrome_danbooru',
    )
