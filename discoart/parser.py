import os

os.environ['DISCOART_DISABLE_IPYTHON'] = '1'  # turn on when using from CLI

import argparse
import sys

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
    sps = parser.add_subparsers(dest='cli', required=True)
    sp1 = sps.add_parser('create', help='Create artworks from a YAML config file')
    sp1.add_argument(
        'input',
        metavar='YAML_CONFIG_FILE',
        nargs='?',
        type=argparse.FileType('r'),
        help='The YAML config file to use, default is stdin.',
        default=sys.stdin,
    )
    sp2 = sps.add_parser('config', help='Export the default config to a YAML file')
    sp2.add_argument(
        'output',
        nargs='?',
        metavar='EXPORT_YAML_FILE',
        type=argparse.FileType('w'),
        help='The file path to export to, default is stdout.',
        default=sys.stdout,
    )
    sp3 = sps.add_parser(
        'serve', help='Serve DiscoArt as a gRPC/HTTP/Websocket service'
    )
    sp3.add_argument(
        'input',
        nargs='?',
        metavar='FLOW_YAML_FILE',
        type=argparse.FileType('r'),
        help='The Jina Flow YAML config file to use.',
        default=os.path.join(__resources_path__, 'flow.yml'),
    )
    return parser
