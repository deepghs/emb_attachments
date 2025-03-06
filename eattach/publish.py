import datetime
import glob
import json
import os
import re
from contextlib import contextmanager
from functools import partial
from tempfile import TemporaryDirectory
from typing import Optional

import click
import torch
from ditk import logging
from hbutils.encoding import sha3
from hfutils.archive import archive_unpack
from huggingface_hub import HfApi, CommitOperationAdd

from eattach.model import Backbone
from .encode import load_encoder
from .onnx import export_backbone_to_onnx
from .problem import load_problem
from .utils import GLOBAL_CONTEXT_SETTINGS, torch_model_profile
from .utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'publish')

_LOG_FILE_PATTERN = re.compile(r'^events\.out\.tfevents\.(?P<timestamp>\d+)\.(?P<machine>[^.]+)\.(?P<extra>[\s\S]+)$')


@contextmanager
def export_model(workdir: str, logfile_anonymous: bool = True, verbose: bool = True):
    with TemporaryDirectory() as td:
        files = []
        meta_file = os.path.join(workdir, 'meta.json')
        with open(meta_file, 'r') as f:
            meta_info = json.load(f)

        for logfile in glob.glob(os.path.join(workdir, 'events.out.tfevents.*')):
            logging.info(f'Tensorboard file {logfile!r} found.')
            matching = _LOG_FILE_PATTERN.fullmatch(os.path.basename(logfile))
            assert matching, f'Log file {logfile!r}\'s name not match with pattern {_LOG_FILE_PATTERN.pattern}.'

            timestamp = matching.group('timestamp')
            machine = matching.group('machine')
            if logfile_anonymous:
                machine = sha3(machine.encode(), n=224)
            extra = matching.group('extra')

            final_name = f'events.out.tfevents.{timestamp}.{machine}.{extra}'
            files.append((logfile, final_name))

        archive_unpack(
            archive_file=os.path.join(workdir, 'checkpoints', 'best.zip'),
            directory=td,
            silent=True,
        )
        files.append((os.path.join(td, 'metrics.json'), 'metrics.json'))
        backbone_file = os.path.join(td, 'model.safetensors')
        files.append((backbone_file, 'model.safetensors'))
        for plot_file in glob.glob(os.path.join(td, 'plots', '*')):
            files.append((plot_file, f'plot_{os.path.basename(plot_file)}'))

        onnx_file = os.path.join(td, 'model.onnx')
        encoder = load_encoder(model_name=meta_info['encoder_model'])
        problem = load_problem(config=meta_info['problem'])
        export_backbone_to_onnx(
            backbone_file=os.path.join(td, 'model.safetensors'),
            encoder=encoder,
            problem=problem,
            onnx_filename=onnx_file,
            verbose=verbose,
        )
        files.append((onnx_file, 'model.onnx'))

        backbone, metadata = Backbone.load(backbone_file, with_metadata=True)
        model = backbone.module.float()
        dummy_input = torch.randn(1, encoder.width)
        dummy_input = dummy_input.float()
        flops, params = torch_model_profile(model=model, input_=dummy_input)
        meta_info['flops'] = flops
        meta_info['params'] = params

        new_meta_file = os.path.join(td, 'meta.json')
        with open(new_meta_file, 'w') as f:
            json.dump(meta_info, f, indent=4, sort_keys=True, ensure_ascii=False)
        files.append((new_meta_file, 'meta.json'))

        yield files


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Utils publishing models.")
def cli():
    pass  # pragma: no cover


@cli.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Publish model to huggingface model repository')
@click.option('--workdir', '-w', 'workdir', type=click.Path(file_okay=False, exists=True), required=True,
              help='Work directory of the training.', show_default=True)
@click.option('--verbose', '-V', 'verbose', is_flag=True, type=bool, default=False,
              help='Show verbose information.', show_default=True)
@click.option('--name', '-n', 'name', type=str, default=None,
              help='Name of the checkpoint. Default is the basename of the work directory.', show_default=True)
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository for publishing model.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
def huggingface(workdir: str, verbose: bool, name: Optional[str], repository: str, revision: str):
    logging.try_init_root(logging.INFO)

    hf_client = HfApi(token=os.environ['HF_TOKEN'])
    logging.info(f'Initialize repository {repository!r}')
    if not hf_client.repo_exists(repo_id=repository, repo_type='model'):
        hf_client.create_repo(repo_id=repository, repo_type='model', exist_ok=True)

    name = name or os.path.basename(os.path.abspath(workdir))
    with export_model(workdir, verbose=verbose) as files:
        current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        commit_message = f"Publish model {name}, on {current_time}"
        logging.info(f'Publishing model {name!r} to repository {repository!r} ...')
        hf_client.create_commit(
            repository,
            [
                CommitOperationAdd(
                    path_in_repo=f'{name}/{filename}',
                    path_or_fileobj=local_file,
                ) for local_file, filename in files
            ],
            commit_message=commit_message,
            repo_type='model',
            revision=revision,
        )


if __name__ == '__main__':
    cli()
