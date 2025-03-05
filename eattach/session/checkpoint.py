import json
import os
import zipfile
from typing import Optional, Mapping, Union, Dict, Any

from PIL import Image
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.archive import archive_pack

from .base import BaseLogger
from ..model import Backbone


class CheckpointLogger(BaseLogger):
    def __init__(self, workdir: str, key_metric: str = 'accuracy',
                 extra_metadata: Optional[Dict[str, Any]] = None, **kwargs):
        BaseLogger.__init__(self, workdir, **kwargs)
        self.key_metric = key_metric
        self.extra_metadata = dict(extra_metadata or {})

        self.ckpt_dir = os.path.join(self.workdir, 'checkpoints')
        self._last_step: Optional[int] = None
        self._best_metric_value: Optional[float] = None
        self._load_last()
        self._load_best()

    @classmethod
    def _load_ckpt(cls, ckpt_file: str):
        with TemporaryDirectory() as td, zipfile.ZipFile(ckpt_file, 'r') as zf:
            meta_json_file = os.path.join(td, 'meta.json')
            zf.extract('meta.json', td)
            with open(meta_json_file, 'r') as f:
                meta = json.load(f)

            metrics_json_file = os.path.join(td, 'metrics.json')
            zf.extract('metrics.json', td)
            with open(metrics_json_file, 'r') as f:
                metrics = json.load(f)

            return meta, metrics

    def _save_ckpt(self, ckpt_file: str, global_step, model: Backbone,
                   metrics: Mapping[str, Union[float, Image.Image]]):
        with TemporaryDirectory() as td:
            with open(os.path.join(td, 'meta.json'), 'w') as f:
                json.dump({
                    'step': global_step,
                    'model_type': model.type,
                    **model.init_params,
                }, f, indent=4, sort_keys=True, ensure_ascii=False)
            with open(os.path.join(td, 'metrics.json'), 'w') as f:
                json.dump({
                    key: value for key, value in metrics.items()
                    if not isinstance(value, Image.Image)
                }, f, indent=4, sort_keys=True, ensure_ascii=False)
            model.save(
                ckpt_file=os.path.join(td, 'model.safetensors'),
                extra_metadata={
                    **self.extra_metadata,
                    **{
                        f'metrics/{key}': value for key, value in metrics.items()
                        if not isinstance(value, Image.Image)
                    }
                }
            )
            for key, value in metrics.items():
                if isinstance(value, Image.Image):
                    plot_file = os.path.join(td, 'plots', f'{key}.png')
                    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
                    value.save(plot_file)

            if os.path.dirname(ckpt_file):
                os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
            archive_pack('zip', directory=td, archive_file=ckpt_file)

    @property
    def _last_ckpt(self):
        return os.path.join(self.ckpt_dir, 'last.zip')

    def _load_last(self):
        if os.path.exists(self._last_ckpt):
            meta, metrics = self._load_ckpt(self._last_ckpt)
            self._last_step = meta['step']
            logging.info(f'Last ckpt found at {self._last_step}, with previous step {self._last_step}')
        else:
            self._last_step = None
            logging.info('No last ckpt found.')

    def _save_last(self, global_step, model: Backbone, metrics: Mapping[str, Union[float, Image.Image]]):
        self._save_ckpt(
            ckpt_file=self._last_ckpt,
            global_step=global_step,
            model=model,
            metrics=metrics,
        )
        self._last_step = global_step
        logging.info(f'Last ckpt model epoch {global_step} saved')

    @property
    def _best_ckpt(self):
        return os.path.join(self.ckpt_dir, 'best.zip')

    def _load_best(self):
        if os.path.exists(self._best_ckpt):
            meta, metrics = self._load_ckpt(self._best_ckpt)
            self._best_metric_value = metrics[self.key_metric]
            step = meta['step']
            logging.info(f'Best ckpt found at {self._best_ckpt}, '
                         f'with step {step} and {self.key_metric} {self._best_metric_value:.3f}')
        else:
            self._best_metric_value = None
            logging.info('No best ckpt found.')

    def _save_best(self, global_step, model: Backbone, metrics: Mapping[str, Union[float, Image.Image]]):
        if self._best_metric_value is None or metrics[self.key_metric] > self._best_metric_value:
            self._save_ckpt(
                ckpt_file=self._best_ckpt,
                global_step=global_step,
                model=model,
                metrics=metrics,
            )
            self._best_metric_value = metrics[self.key_metric]
            logging.info(f'Best ckpt model epoch {global_step} saved, '
                         f'with {self.key_metric}\'s new value {self._best_metric_value:.3f}')

    def tb_train_log(self, global_step, metrics: Mapping[str, Union[float, Image.Image]]):
        pass

    def tb_eval_log(self, global_step, model: Backbone, metrics: Mapping[str, Union[float, Image.Image]]):
        self._save_last(global_step, model, metrics)
        self._save_best(global_step, model, metrics)
