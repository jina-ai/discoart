import copy
import random
import warnings
from types import SimpleNamespace
from typing import Dict

import yaml
from yaml import Loader

from . import __resources_path__

with open(f'{__resources_path__}/default.yml') as ymlfile:
    default_args = yaml.load(ymlfile, Loader=Loader)


def load_config(
        user_config: Dict,
):
    cfg = copy.deepcopy(default_args)

    if user_config:
        cfg.update(**user_config)

    for k in user_config.keys():
        if k not in cfg:
            warnings.warn(f'unknown argument {k}, ignored')

    for k, v in cfg.items():
        if k in ('batch_size', 'display_rate', 'seed', 'skip_steps', 'steps', 'n_batches', 'cutn_batches') and isinstance(v, float):
            cfg[k] = int(v)
        if k == 'width_height':
            cfg[k] = [int(vv) for vv in v]

    cfg.update(
        **{
            'seed': cfg['seed'] or random.randint(0, 2 ** 32),
        }
    )

    if cfg['batch_name']:
        da_name = f'{__package__}-{cfg["batch_name"]}-{cfg["seed"]}'
    else:
        da_name = f'{__package__}-{cfg["seed"]}'
        warnings.warn('you did not set `batch_name`, set it to have unique session ID')

    cfg.update(
        **{
            'name_docarray': da_name
        }
    )

    print_args_table(cfg)

    return SimpleNamespace(**cfg)


def print_args_table(cfg):
    from rich.table import Table
    from rich import box
    from rich.console import Console

    console = Console()

    param_str = Table(
        title=cfg['name_docarray'],
        box=box.ROUNDED,
        highlight=True,
        title_justify='left',
    )
    param_str.add_column('Argument', justify='right')
    param_str.add_column('Value', justify='left')

    for k, v in sorted(cfg.items()):
        value = str(v)

        if not default_args.get(k, None) == v:
            value = f'[b]{value}[/]'

        param_str.add_row(k, value)

    console.print(param_str)
