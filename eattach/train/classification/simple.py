import json
import os
from typing import Optional

from accelerate import Accelerator
from ditk import logging
from hbutils.random import global_seed
from torch.utils.data import Dataset

from eattach.model import Backbone


def train_classification(
        workdir: str, model_type: str = 'mlp', init_params: dict = None,
        encode_system: str = 'imgutils.tagging.wd14', model_name: str = 'SwinV2_v3',
        dataset_dir: Optional[str] = None, dataset: Optional[Dataset] = None,
        train_dataset: Optional[Dataset] = None, test_dataset: Optional[Dataset] = None,
        batch_size: int = 16, seed: Optional[int] = 0,
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
        last_step = metadata['train/step']
        logging.info(f'Resume from step {last_step!r}.')
    else:
        logging.info(f'No last checkpoint found, initialize {model_type!r} model with params {init_params!r}.')
        backbone = Backbone.new(type_=model_type, **init_params)
        last_step = 0
    model_type, init_params = backbone.type, backbone.init_params

    train_cfg = {
        'batch_size': batch_size,
    }
    with open(os.path.join(workdir, 'meta.json'), 'w') as f:
        json.dump({
            'problem': 'classification',
            'model_type': model_type,
            'init_params': init_params,
        }, f, indent=4, ensure_ascii=False, sort_keys=True)

    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )
