import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

__version__ = '0.12.0'

__all__ = ['create', 'cheatsheet']

import sys

__resources_path__ = os.path.join(
    os.path.dirname(
        sys.modules.get(__package__).__file__
        if __package__ in sys.modules
        else __file__
    ),
    'resources',
)

from .create import create, gobig
from .config import cheatsheet, show_config, save_config, load_config
