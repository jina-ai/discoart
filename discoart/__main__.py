import os

os.environ['DISCOART_DISABLE_IPYTHON'] = '1'  # turn on when using from CLI
import argparse

import sys

from yaml import Loader
import yaml
from . import __version__, __resources_path__


def get_main_parser():
    parser = argparse.ArgumentParser(
        epilog=f'Create compelling Disco Diffusion artworks in one line',
        prog='python -m discoart',
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=__version__,
        help='Show DiscoArt version',
    )
    sps = parser.add_subparsers(dest='cli_command')
    sp1 = sps.add_parser('create', help='Create artworks from a YAML config file')
    sp1.add_argument(
        'input',
        nargs='?',
        type=argparse.FileType('r'),
        help='The YAML config file to use, default is stdin.',
        default=sys.stdin,
    )
    sp2 = sps.add_parser('config', help='Export the default config to a YAML file')
    sp2.add_argument(
        'output',
        nargs='?',
        type=argparse.FileType('w'),
        help='The file path to export to, default is stdout.',
        default=sys.stdout,
    )
    return parser


if __name__ == '__main__':
    parser = get_main_parser()
    args = parser.parse_args()
    if args.cli_command == 'config':
        with open(
            os.environ.get(
                'DISCOART_DEFAULT_PARAMETERS_YAML', f'{__resources_path__}/default.yml'
            )
        ) as fp:
            args.output.write(fp.read())
            args.output.close()
    elif args.cli_command == 'create':
        kwargs = yaml.load(args.input, Loader=Loader)
        from .create import create

        create(**kwargs)
