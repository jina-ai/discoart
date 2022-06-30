import copy
import random
from types import SimpleNamespace
from typing import Dict

import yaml
from yaml import Loader

from . import __resources_path__


def load_config(
    user_config: Dict, default_config: str = f'{__resources_path__}/default.yml'
):
    with open(default_config) as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
        default_args = copy.deepcopy(cfg)

    if user_config:
        cfg.update(**user_config)

    cfg.update(
        **{
            'seed': cfg['seed'] or random.randint(0, 2**32),
        }
    )

    cfg.update(
        **{
            'name_docarray': f'{__package__}-{cfg["seed"]}',
        }
    )

    print_args_table(cfg, default_args)

    return SimpleNamespace(**cfg)


def print_args_table(cfg, default_cfg):
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

        if not default_cfg.get(k, None) == v:
            value = f'[b]{value}[/]'

        param_str.add_row(k, value)

    console.print(param_str)
