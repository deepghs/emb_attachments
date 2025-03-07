import json
import os
import random
from typing import Optional, Callable

import torch
from accelerate import Accelerator
from ditk import logging
from hbutils.random import global_seed
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ..metrics import cls_map_score, cls_auc_score
from ...dataset import ImageDirDataset, dataset_split, WrappedImageDataset, load_labels_from_image_dir, RangeRandomCrop
from ...encode import load_encoder, EncoderModel
from ...model import Backbone, TrainModel
from ...plot import plt_export, plt_confusion_matrix, plt_p_curve, plt_r_curve, plt_pr_curve, plt_f1_curve, \
    plt_roc_curve
from ...problem import ClassificationProblem
from ...session import TrainSession

DEFAULT_INIT_PARAMS = {
    'layers': [0.5],
}


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
        key_metric: str = 'AUC',
        loss: str = 'focal',
        seed: Optional[int] = 0,
        eval_epoch: int = 1,
):
    if seed is None:
        seed = random.randint(0, (1 << 31) - 1)
    # native random, numpy, torch and faker's seeds are includes
    # if you need to register more library for seeding, see:
    # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
    logging.info(f'Globally set the random seed {seed!r}.')
    global_seed(seed)

    os.makedirs(workdir, exist_ok=True)

    labels_info = load_labels_from_image_dir(dataset_dir, unsupervised=None)
    logging.info(f'Load labels from dataset directory {dataset_dir!r}, '
                 f'labels: {labels_info.labels!r}, unsupervised: {labels_info.unsupervised!r}.')
    problem = ClassificationProblem(labels=labels_info.labels)
    encoder: EncoderModel = load_encoder(encoder_model)

    init_params = dict(init_params if init_params is not None else DEFAULT_INIT_PARAMS)
    init_params['in_dims'] = encoder.width
    init_params['out_dims'] = len(labels_info.labels)
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
        previous_epoch = metadata['step']
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

    train_cfg = {
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'seed': seed,
        'loss': loss,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'key_metric': key_metric,
        'processes': accelerator.num_processes,
    }
    logging.info(f'Training configurations: {train_cfg!r}.')
    with open(os.path.join(workdir, 'meta.json'), 'w') as f:
        json.dump({
            'problem': problem.to_json(),
            'model_type': model_type,
            'init_params': init_params,
            'encoder_model': encoder_model,
            'train': train_cfg,
        }, f, indent=4, ensure_ascii=False, sort_keys=True)

    dataset = ImageDirDataset(
        dataset_dir,
        labels=labels_info.labels,
        no_cache=True,
    )
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
    logging.info(f'Backbone structure:\n{backbone.module}')

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
        session = TrainSession(
            workdir, key_metric=key_metric,
            extra_metadata={
                **{f'train/{key}': value for key, value in train_cfg.items()},
                **{f'problem/{key}': value for key, value in problem.to_json().items()},
                'encoder_model': encoder_model,
            }
        )
        logging.info('Training start!')

    for epoch in range(previous_epoch + 1, max_epochs + 1):
        model.train()
        train_lr = scheduler.get_last_lr()[0]
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
            train_y_true = train_y_true.detach().cpu().numpy()
            train_y_pred = train_y_pred.detach().cpu().numpy()
            train_y_score = train_y_score.detach().cpu().numpy()

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
                    'learning_rate': train_lr,
                }
            )

        if epoch % eval_epoch == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_total = 0
                test_y_true, test_y_pred, test_y_score = [], [], []

                for i, (inputs, labels_) in enumerate(tqdm(test_dataloader, disable=not accelerator.is_main_process)):
                    inputs = inputs.float()
                    labels_ = labels_.long()

                    outputs = model(inputs)
                    test_y_true.append(labels_)
                    test_y_pred.append(outputs.argmax(dim=1))
                    test_y_score.append(torch.softmax(outputs, dim=1))
                    test_total += labels_.shape[0]

                    loss = loss_fn(outputs, labels_)
                    test_loss += loss.item() * inputs.size(0)

                test_y_true = accelerator.gather_for_metrics(torch.concat(test_y_true))
                test_y_pred = accelerator.gather_for_metrics(torch.concat(test_y_pred))
                test_y_score = accelerator.gather_for_metrics(torch.concat(test_y_score))

                if accelerator.is_main_process:
                    test_y_true = test_y_true.detach().cpu().numpy()
                    test_y_pred = test_y_pred.detach().cpu().numpy()
                    test_y_score = test_y_score.detach().cpu().numpy()

                    session.tb_eval_log(
                        global_step=epoch,
                        model=backbone,
                        metrics={
                            'loss': test_loss / test_total,
                            'accuracy': accuracy_score(test_y_true, test_y_pred),
                            'mAP': cls_map_score(test_y_true, test_y_score, problem.labels),
                            'AUC': cls_auc_score(test_y_true, test_y_score, problem.labels),
                            'confusion': plt_export(
                                plt_confusion_matrix,
                                test_y_true, test_y_pred, problem.labels,
                                title=f'Test Confusion Epoch {epoch}',
                                figsize=(cm_size, cm_size),
                            ),
                            'p_curve': plt_export(
                                plt_p_curve, test_y_true, test_y_score, problem.labels,
                                title=f'Precision Epoch {epoch}',
                            ),
                            'r_curve': plt_export(
                                plt_r_curve, test_y_true, test_y_score, problem.labels,
                                title=f'Recall Epoch {epoch}',
                            ),
                            'pr_curve': plt_export(
                                plt_pr_curve, test_y_true, test_y_score, problem.labels,
                                title=f'PR Curve Epoch {epoch}',
                            ),
                            'f1_curve': plt_export(
                                plt_f1_curve, test_y_true, test_y_score, problem.labels,
                                title=f'F1 Curve Epoch {epoch}',
                            ),
                            'roc_curve': plt_export(
                                plt_roc_curve, test_y_true, test_y_score, problem.labels,
                                title=f'ROC Curve Epoch {epoch}',
                            ),
                        }
                    )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    seed = int(os.environ.get('SEED', 0))
    train_classification(
        workdir=f'runs/train_test_1lx_s{seed}',
        dataset_dir='/data/monochrome_danbooru',
        train_augment=transforms.Compose([
            transforms.Resize((500, 500)),
            RangeRandomCrop((400, 500), padding=0, pad_if_needed=True, padding_mode='reflect'),
            transforms.RandomRotation((-45, 45)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.10, 0.10, 0.05, 0.03),
        ]),
        seed=seed,
    )
