import io
import json
import os
import re
from functools import partial
from tempfile import TemporaryDirectory

import click
import numpy as np
import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hfutils.operate import upload_directory_as_directory
from hfutils.utils import hf_fs_path
from huggingface_hub import HfFileSystem, HfApi
from huggingface_hub.hf_api import RepoFile
from thop import clever_format
from tqdm import tqdm

from .problem import load_problem, ClassificationProblem, TaggingProblem, RegressionProblem
from .utils import GLOBAL_CONTEXT_SETTINGS, markdown_to_df
from .utils import print_version as _origin_print_version

print_version = partial(_origin_print_version, 'list')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="List published models.")
def cli():
    pass  # pragma: no cover


def _name_process(name: str):
    words = re.split(r'[\W_]+', name)
    return ' '.join([
        word.capitalize() if re.fullmatch('^[a-z0-9]+$', word) else word
        for word in words
    ])


_PERCENTAGE_METRICS = ('accuracy',)


@cli.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Publish model to huggingface model repository')
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository for publishing model.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
def huggingface(repository: str, revision: str):
    logging.try_init_root(logging.INFO)

    hf_fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))
    hf_client = HfApi(token=os.environ.get('HF_TOKEN'))

    names = [fn.split('/')[-2] for fn in hf_fs.glob(f'{repository}@{revision}/*/model.safetensors')]
    logging.info(f'{plural_word(len(names), "model")} detected in {repository}@{revision}')

    rows = []
    for name in tqdm(names):
        meta_info = json.loads(hf_fs.read_text(f'{repository}@{revision}/{name}/meta.json'))
        row = {'Name': name}

        problem = load_problem(meta_info['problem'])
        row['Encoder'] = meta_info['encoder_model']
        row['Problem'] = meta_info['problem']['type']

        flops, params = meta_info['flops'], meta_info['params']
        row['FLOPS'], row['Params'] = clever_format([flops, params], "%.1f")

        repo_file: RepoFile = list(hf_client.get_paths_info(
            repo_id=repository,
            repo_type='model',
            paths=[f'{name}/model.safetensors'],
            expand=True,
        ))[0]
        last_commit_at = repo_file.last_commit.date.timestamp()

        with open(hf_client.hf_hub_download(repository, f'{name}/metrics.json', revision=revision), 'r') as f:
            metrics = json.load(f)
            for metric_name, metric_value in metrics.items():
                row[metric_name] = f'{metric_value:.4g}'

        if isinstance(problem, ClassificationProblem):
            row['Labels'] = ', '.join(problem.labels)
        elif isinstance(problem, TaggingProblem):
            row['Tags'] = ', '.join(problem.tags)
        elif isinstance(problem, RegressionProblem):
            row['Values'] = ', '.join([name for name, _, _ in problem.fields])

        row['created_at'] = last_commit_at
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['created_at'], ascending=[False])
    del df['created_at']
    df = df.replace(np.nan, 'N/A')

    with TemporaryDirectory() as td:
        with open(os.path.join(td, 'README.md'), 'w') as f:
            if not hf_fs.exists(hf_fs_path(
                    repo_id=repository,
                    repo_type='model',
                    filename='README.md',
                    revision=revision,
            )):
                print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

            else:
                table_printed = False
                tb_lines = []
                with io.StringIO(hf_fs.read_text(hf_fs_path(
                        repo_id=repository,
                        repo_type='model',
                        filename='README.md',
                        revision=revision,
                )).rstrip() + os.linesep * 2) as ifx:
                    for line in ifx:
                        line = line.rstrip()
                        if line.startswith('|') and not table_printed:
                            tb_lines.append(line)
                        else:
                            if tb_lines:
                                df_c = markdown_to_df(os.linesep.join(tb_lines))
                                if 'Name' in df_c.columns and 'FLOPS' in df_c.columns and 'Params' in df_c.columns:
                                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)
                                    table_printed = True
                                    tb_lines.clear()
                                else:
                                    print(os.linesep.join(tb_lines), file=f)
                            print(line, file=f)

                if not table_printed:
                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='model',
            revision=revision,
            path_in_repo='.',
            local_directory=td,
            message=f'Sync README for {repository}',
            hf_token=os.environ.get('HF_TOKEN'),
        )


if __name__ == '__main__':
    cli()
