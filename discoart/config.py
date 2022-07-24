import copy
import os
import random
from typing import Dict, Union, Optional

import yaml
from docarray import DocumentArray, Document
from yaml import Loader

from . import __resources_path__

with open(
    os.environ.get(
        'DISCOART_DEFAULT_PARAMETERS_YAML',
        os.path.join(__resources_path__, 'default.yml'),
    )
) as ymlfile:
    default_args = yaml.load(ymlfile, Loader=Loader)

with open(
    os.environ.get(
        'DISCOART_CUT_SCHEDULES_YAML',
        os.path.join(__resources_path__, 'cut-schedules.yml'),
    )
) as ymlfile:
    cut_schedules = yaml.load(ymlfile, Loader=Loader)


def load_config(
    user_config: Dict,
) -> Dict:
    cfg = copy.deepcopy(default_args)

    for k in list(user_config.keys()):
        if k not in cfg and not k.startswith('_'):
            raise AttributeError(f'unknown argument `{k}`, misspelled?')
        if k.startswith('_'):
            # remove private arguments in tags
            user_config.pop(k)

    if user_config:
        if user_config.get('cut_schedules_group', None):
            cfg.update(cut_schedules[user_config['cut_schedules_group']])

        cfg.update(**user_config)

    for k, v in cfg.items():
        if k in (
            'batch_size',
            'display_rate',
            'seed',
            'skip_steps',
            'steps',
            'n_batches',
            'cutn_batches',
        ) and isinstance(v, float):
            cfg[k] = int(v)
        if k == 'width_height':
            cfg[k] = [int(vv) for vv in v]

    cfg.update(
        **{
            'seed': cfg['seed'] or random.randint(0, 2**32),
        }
    )

    _id = random.getrandbits(128).to_bytes(16, 'big').hex()
    if cfg['batch_name']:
        da_name = f'{__package__}-{cfg["batch_name"]}-{_id}'
    else:
        da_name = f'{__package__}-{_id}'
        from .helper import logger

        logger.debug('you did not set `batch_name`, set it to have unique session ID')

    if not cfg.get('name_docarray', None):
        cfg['name_docarray'] = da_name

    return cfg


def show_config(
    docs: Union['Document', 'Document', Dict, str], only_non_default: bool = False
):
    cfg = None

    if isinstance(docs, DocumentArray):
        cfg = docs[0].tags
    elif isinstance(docs, Document):
        cfg = docs.tags
    elif isinstance(docs, dict):
        cfg = docs
    elif isinstance(docs, str):
        cfg = DocumentArray.pull(docs)[0].tags
    print_args_table(cfg, only_non_default=only_non_default)


def save_config_svg(
    docs: Union['DocumentArray', 'Document', Dict],
    output: Optional[str] = None,
) -> None:
    """
    Save the config as SVG.
    :param docs: a DocumentArray or Document or a Document.tags dict
    :param output: the filename to store the SVG, if not given, it will be saved as `{name_docarray}.svg`
    :return:
    """
    cfg = None

    if isinstance(docs, DocumentArray):
        cfg = docs[0].tags
    elif isinstance(docs, Document):
        cfg = docs.tags
    elif isinstance(docs, dict):
        cfg = docs

    from rich.console import Console
    from rich.terminal_theme import MONOKAI

    console = Console(record=True)
    cfg = load_config(cfg)
    print_args_table(cfg, console)
    console.save_svg(
        output or f'{cfg["name_docarray"]}.svg',
        theme=MONOKAI,
        title=cfg['name_docarray'],
    )


def print_args_table(
    cfg, console=None, only_non_default: bool = False, console_print: bool = True
):
    from rich.table import Table
    from rich import box
    from rich.console import Console

    if console is None:
        console = Console()

    param_str = Table(
        title=cfg['name_docarray'],
        caption=f'showing only non-default args'
        if only_non_default
        else 'showing all args ([b]bold *[/] args are non-default)',
        box=box.ROUNDED,
        highlight=True,
        title_justify='center',
    )
    param_str.add_column('Argument', justify='right')
    param_str.add_column('Value', justify='left')

    for k, v in sorted(cfg.items()):
        if k.startswith('_'):
            continue
        value = str(v)
        _non_default = False
        if not default_args.get(k, None) == v:
            if not only_non_default:
                k = f'[b]{k}*[/]'
            _non_default = True

        if not only_non_default or _non_default:
            param_str.add_row(k, value)

    if console_print:
        console.print(param_str)
    return param_str


def cheatsheet():
    from . import __resources_path__

    from rich.table import Table
    from rich import box
    from rich.console import Console

    console = Console()

    with open(os.path.join(__resources_path__, 'docstrings.yml')) as ymlfile:
        docs = yaml.load(ymlfile, Loader=Loader)

    param_tab = Table(
        title=f'Cheatsheet for all supported parameters',
        box=box.ROUNDED,
        highlight=True,
        show_lines=True,
        title_justify='center',
    )
    param_tab.add_column('Argument', justify='right')
    param_tab.add_column('Default', justify='left', max_width=10, overflow='fold')
    param_tab.add_column('Description', justify='left')

    for k, v in sorted(default_args.items()):
        value = str(v)
        if k in docs:
            d_string = docs[k]
            param_tab.add_row(
                str(k),
                value,
                d_string.replace('[DiscoArt]', '[bold white on red]:new: DiscoArt[/]'),
            )

    console.print(param_tab)
