import os

os.environ['DISCOART_DISABLE_IPYTHON'] = '1'  # turn on when using from CLI

if __name__ == '__main__':
    from .parser import get_main_parser

    args = get_main_parser().parse_args()

    from . import api

    getattr(api, args.cli.replace('-', '_'))(args)
