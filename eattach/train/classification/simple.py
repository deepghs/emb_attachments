import json
import os
from typing import Optional, Callable

import torch
from accelerate import Accelerator
from ditk import logging
from hbutils.random import global_seed
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from eattach.plot import plt_export, plt_confusion_matrix
from eattach.session import TrainSession
from eattach.train.metrics import cls_map_score, cls_auc_score
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
        num_workers: int = 8,
        train_augment: Optional[Callable] = None,
        test_split_ratio: float = 0.2,

        # train hyperparams
        max_epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-3,
        key_metric: str = 'accuracy',
        loss: str = 'focal',
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
        previous_epoch = metadata['train/epoch']
        logging.info(f'Resume from epoch {previous_epoch!r}.')
    else:
        logging.info(f'No last checkpoint found, initialize {model_type!r} model with params {init_params!r}.')
        backbone = Backbone.new(type_=model_type, **init_params)
        previous_epoch = 0
    model_type, init_params = backbone.type, backbone.init_params

    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    encoder: EncoderModel = load_encoder(encoder_model)

    train_cfg = {
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'seed': seed,
        'loss': loss,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'key_metric': key_metric,
    }
    with open(os.path.join(workdir, 'meta.json'), 'w') as f:
        json.dump({
            'problem': 'classification',
            'model_type': model_type,
            'init_params': init_params,
            'encoder_model': encoder_model,
            'train': train_cfg,
        }, f, indent=4, ensure_ascii=False, sort_keys=True)

    labels_info = load_labels_from_image_dir(dataset_dir, unsupervised=None)
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

    cm_size = max(6.0, len(labels_info.labels) * 0.9)

    model = TrainModel(
        encoder=encoder.model,
        backbone=backbone.module,
        head=None,
    )
    logging.info(f'Model structure:\n{model}')

    num_workers = num_workers or min(os.cpu_count(), batch_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    loss_fn = problem.get_loss_fn(loss_fn_name=loss, reduction='mean')
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader), epochs=max_epochs,
        pct_start=0.15, final_div_factor=20.
    )

    model, optimizer, train_dataloader, test_dataloader, scheduler, loss_fn = \
        accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler, loss_fn)

    if accelerator.is_main_process:
        session = TrainSession(workdir, key_metric=key_metric)
        logging.info('Training start!')

    for epoch in range(previous_epoch + 1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_y_true, train_y_pred, train_y_score = [], [], []

        for i, (inputs, labels_) in enumerate(tqdm(train_dataloader, disable=not accelerator.is_main_process)):
            inputs = inputs.float()
            labels_ = labels_

            optimizer.zero_grad()
            outputs = model(inputs)
            train_y_true.append(labels_)
            train_y_pred.append(outputs.argmax(dim=1))
            train_y_score.append(torch.softmax(outputs, dim=1))
            train_total += labels_.shape[0]

            print(outputs.shape, labels_.shape)

            loss = loss_fn(outputs, labels_)
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            scheduler.step()

        # Gather results from all processes
        train_y_true = accelerator.gather(torch.concat(train_y_true))
        train_y_pred = accelerator.gather(torch.concat(train_y_pred))
        train_y_score = accelerator.gather(torch.concat(train_y_score))

        if accelerator.is_main_process:
            train_y_true = train_y_true.cpu().numpy()
            train_y_pred = train_y_pred.cpu().numpy()
            train_y_score = train_y_score.cpu().numpy()

            session.tb_train_log(
                global_step=epoch,
                metrics={
                    'loss': train_loss / train_total,
                    'accuracy': accuracy_score(train_y_true, train_y_pred),
                    'mAP': cls_map_score(train_y_true, train_y_score, problem.labels),
                    'AUC': cls_auc_score(train_y_true, train_y_score, problem.labels),
                    'confusion': plt_export(
                        plt_confusion_matrix,
                        train_y_true, train_y_pred, problem.labels,
                        title=f'Train Confusion Epoch {epoch}',
                        figsize=(cm_size, cm_size),
                    ),
                }
            )

        break


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    train_classification(
        workdir='runs/train_test',
        dataset_dir='/data/monochrome_danbooru',
    )
