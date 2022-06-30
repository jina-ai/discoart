import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

__version__ = '0.0.1'

__all__ = ['create']

import sys

__resources_path__ = os.path.join(
    os.path.dirname(
        sys.modules.get(__package__).__file__
        if __package__ in sys.modules
        else __file__
    ),
    'resources',
)

import gc

# check if GPU is available
import torch

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    raise RuntimeError(
        'CUDA is not available. DiscoArt is unbearably slow on CPU. '
        'Please switch to GPU device, if you are using Google Colab, then free tier would work.'
    )

# download and load models, this will take some time on the first load

from .helper import load_all_models, load_diffusion_model

model_config, clip_models, secondary_model = load_all_models(
    '512x512_diffusion_uncond_finetune_008100',
    use_secondary_model=True,
    device=device,
)

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from docarray import DocumentArray


# begin_create_overload

# end_create_overload


def create(**kwargs) -> 'DocumentArray':
    from .config import load_config
    from .runner import do_run

    _args = load_config(user_config=kwargs)

    model, diffusion = load_diffusion_model(
        model_config, _args.diffusion_model, steps=_args.steps, device=device
    )

    gc.collect()
    torch.cuda.empty_cache()
    try:
        return do_run(_args, (model, diffusion, clip_models, secondary_model), device)
    except KeyboardInterrupt:
        pass
    finally:
        from rich import print
        from rich.markdown import Markdown

        md = Markdown(
            f'''
Results are stored in a [DocumentArray](https://docarray.jina.ai/fundamentals/documentarray/).

To get it on any machine:

```python
from docarray import DocumentArray

da = DocumentArray.pull('{_args.name_docarray}')
```

More usage such as plotting, post-analysis can be found  in the [README](https://github.com/jina-ai/discoart).
        '''
        )
        print(md)
        gc.collect()
        torch.cuda.empty_cache()

