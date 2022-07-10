import hashlib
import logging
import os
import subprocess
import sys
from os.path import expanduser
from pathlib import Path
from typing import Dict, Any, List, Tuple

import regex as re
import torch
from open_clip import SimpleTokenizer
from open_clip.tokenizer import whitespace_clean, basic_clean
from spellchecker import SpellChecker

cache_dir = f'{expanduser("~")}/.cache/{__package__}'


def is_jupyter() -> bool:  # pragma: no cover
    """
    Check if we're running in a Jupyter notebook, using magic command `get_ipython` that only available in Jupyter.

    :return: True if run in a Jupyter notebook else False.
    """
    if 'DISCOART_DISABLE_IPYTHON' in os.environ:
        return False

    try:
        get_ipython  # noqa: F821
    except NameError:
        return False
    shell = get_ipython().__class__.__name__  # noqa: F821
    if shell == 'ZMQInteractiveShell':
        return True  # Jupyter notebook or qtconsole
    elif shell == 'Shell':
        return True  # Google colab
    elif shell == 'TerminalInteractiveShell':
        return False  # Terminal running IPython
    else:
        return False  # Other type (?)


def get_ipython_funcs():
    class NOP:
        def __call__(self, *args, **kwargs):
            return NOP()

        __getattr__ = __enter__ = __exit__ = __call__

    if is_jupyter():
        from IPython import display as dp1
        from IPython.display import display as dp2
        from IPython.display import FileLink as fl
        from ipywidgets import Output

        return dp1, dp2, fl, Output
    else:
        return NOP(), NOP(), NOP(), NOP()


def _get_logger():
    logger = logging.getLogger(__package__)
    _log_level = os.environ.get('DISCOART_LOG_LEVEL', 'INFO')
    logger.setLevel(_log_level)
    ch = logging.StreamHandler()
    ch.setLevel(_log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.propagate = False
    return logger


logger = _get_logger()

if not os.path.exists(cache_dir):
    logger.info(
        '''
Looks like you are running {__package__} for the first time. In the first time of usage, it will take some time to download all dependencies and models.
You wont see this message on the second run. From the second run, `from discoart import create` will instantly return.
'''
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

logger.debug(f'`.cache` dir is set to: {cache_dir}')

check_model_SHA = False


def _clone_repo_install(repo_url, repo_dir, commit_hash):
    if os.path.exists(repo_dir):
        res = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, cwd=repo_dir
        ).stdout.decode('utf-8')
        logger.debug(f'commit hash: {res}')
        if res.strip() == commit_hash:
            logger.debug(f'{repo_dir} is already cloned and up to date')
            sys.path.append(repo_dir)
            return

        import shutil

        shutil.rmtree(repo_dir)

    res = subprocess.run(
        ['git', 'clone', '--depth', '1', repo_url, repo_dir], stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    logger.debug(f'cloned {repo_url} to {repo_dir}: {res}')
    sys.path.append(repo_dir)


def _clone_dependencies():
    _clone_repo_install(
        'https://github.com/kostarion/guided-diffusion',
        f'{cache_dir}/guided_diffusion',
        commit_hash='99afa5eb238f32aadaad38ae7107318ec4d987d3',
    )
    _clone_repo_install(
        'https://github.com/assafshocher/ResizeRight',
        f'{cache_dir}/resize_right',
        commit_hash='510d4d5b67dccf4efdee9f311ed42609a71f17c5',
    )


def _wget(url, outputdir):
    res = subprocess.run(
        ['wget', url, '-q', '-P', f'{outputdir}'], stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    logger.debug(res)


def load_clip_models(device, enabled: List[str], clip_models: Dict[str, Any] = {}):
    import open_clip

    # load enabled models
    for k in enabled:
        if k not in clip_models:
            if '::' in k:
                # use open_clip loader
                k1, k2 = k.split('::')
                clip_models[k] = (
                    open_clip.create_model_and_transforms(k1, pretrained=k2)[0]
                    .eval()
                    .requires_grad_(False)
                    .to(device)
                )
            else:
                raise ValueError(
                    f'''
Since v0.1, DiscoArt depends on `open-clip` which supports more CLIP variants and pretrained weights. 
The new names is now a string in the format of `<model_name>::<pretrained_weights_name>`, e.g. 
`ViT-B-32::openai` or `ViT-B-32::laion2b_e16`. The full list of supported models and weights can be found here:
https://github.com/mlfoundations/open_clip#pretrained-model-interface
'''
                )

    # disable not enabled models to save memory
    for k in list(clip_models.keys()):
        if k not in enabled:
            clip_models.pop(k)

    return list(clip_models.values())


def load_all_models(
    diffusion_model,
    use_secondary_model,
    fallback=False,
    device=torch.device('cuda:0'),
):
    _clone_dependencies()
    model_256_downloaded = False
    model_512_downloaded = False
    model_secondary_downloaded = False

    model_256_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'
    model_512_SHA = '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'
    model_secondary_SHA = (
        '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'
    )

    model_256_link = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
    model_512_link = 'https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_link = (
        'https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth'
    )

    model_256_link_fb = (
        'https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt'
    )
    model_512_link_fb = 'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_link_fb = (
        'https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth'
    )

    model_256_path = f'{cache_dir}/256x256_diffusion_uncond.pt'
    model_512_path = f'{cache_dir}/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_path = f'{cache_dir}/secondary_model_imagenet_2.pth'

    if fallback:
        model_256_link = model_256_link_fb
        model_512_link = model_512_link_fb
        model_secondary_link = model_secondary_link_fb
    # Download the diffusion model
    if diffusion_model == '256x256_diffusion_uncond':
        if os.path.exists(model_256_path) and check_model_SHA:
            logger.debug('Checking 256 Diffusion File')
            with open(model_256_path, "rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest()
            if hash == model_256_SHA:
                logger.debug('256 Model SHA matches')
                model_256_downloaded = True
            else:
                logger.debug("256 Model SHA doesn't match, redownloading...")
                _wget(model_256_link, cache_dir)
                if os.path.exists(model_256_path):
                    model_256_downloaded = True
                else:
                    logger.debug('First URL Failed using FallBack')
                    load_all_models(
                        diffusion_model, use_secondary_model, True, device=device
                    )
        elif (
            os.path.exists(model_256_path)
            and not check_model_SHA
            or model_256_downloaded == True
        ):
            logger.debug(
                '256 Model already downloaded, check check_model_SHA if the file is corrupt'
            )
        else:
            _wget(model_256_link, cache_dir)
            if os.path.exists(model_256_path):
                model_256_downloaded = True
            else:
                logger.debug('First URL Failed using FallBack')
                load_all_models(diffusion_model, True, device=device)
    elif diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        if os.path.exists(model_512_path) and check_model_SHA:
            logger.debug('Checking 512 Diffusion File')
            with open(model_512_path, "rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest()
            if hash == model_512_SHA:
                logger.debug('512 Model SHA matches')
                if os.path.exists(model_512_path):
                    model_512_downloaded = True
                else:
                    logger.debug('First URL Failed using FallBack')
                    load_all_models(
                        diffusion_model, use_secondary_model, True, device=device
                    )
            else:
                logger.debug("512 Model SHA doesn't match, redownloading...")
                _wget(model_512_link, cache_dir)
                if os.path.exists(model_512_path):
                    model_512_downloaded = True
                else:
                    logger.debug('First URL Failed using FallBack')
                    load_all_models(
                        diffusion_model, use_secondary_model, True, device=device
                    )
        elif (
            os.path.exists(model_512_path)
            and not check_model_SHA
            or model_512_downloaded == True
        ):
            logger.debug(
                '512 Model already downloaded, check check_model_SHA if the file is corrupt'
            )
        else:
            _wget(model_512_link, cache_dir)
            model_512_downloaded = True
    # Download the secondary diffusion model v2
    if use_secondary_model:
        if os.path.exists(model_secondary_path) and check_model_SHA:
            logger.debug('Checking Secondary Diffusion File')
            with open(model_secondary_path, "rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest()
            if hash == model_secondary_SHA:
                logger.debug('Secondary Model SHA matches')
                model_secondary_downloaded = True
            else:
                logger.debug("Secondary Model SHA doesn't match, redownloading...")
                _wget(model_secondary_link, cache_dir)
                if os.path.exists(model_secondary_path):
                    model_secondary_downloaded = True
                else:
                    logger.debug('First URL Failed using FallBack')
                    load_all_models(
                        diffusion_model, use_secondary_model, True, device=device
                    )
        elif (
            os.path.exists(model_secondary_path)
            and not check_model_SHA
            or model_secondary_downloaded == True
        ):
            logger.debug(
                'Secondary Model already downloaded, check check_model_SHA if the file is corrupt'
            )
        else:
            _wget(model_secondary_link, cache_dir)
            if os.path.exists(model_secondary_path):
                model_secondary_downloaded = True
            else:
                logger.debug('First URL Failed using FallBack')
                load_all_models(
                    diffusion_model, use_secondary_model, True, device=device
                )

    from guided_diffusion.script_util import (
        model_and_diffusion_defaults,
    )

    model_config = model_and_diffusion_defaults()

    if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        model_config.update(
            {
                'attention_resolutions': '32, 16, 8',
                'class_cond': False,
                'diffusion_steps': 1000,  # No need to edit this, it is taken care of later.
                'rescale_timesteps': True,
                'timestep_respacing': 250,  # No need to edit this, it is taken care of later.
                'image_size': 512,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 256,
                'num_head_channels': 64,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_fp16': device.type != 'cpu',
                'use_scale_shift_norm': True,
            }
        )
    elif diffusion_model == '256x256_diffusion_uncond':
        model_config.update(
            {
                'attention_resolutions': '32, 16, 8',
                'class_cond': False,
                'diffusion_steps': 1000,  # No need to edit this, it is taken care of later.
                'rescale_timesteps': True,
                'timestep_respacing': 250,  # No need to edit this, it is taken care of later.
                'image_size': 256,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 256,
                'num_head_channels': 64,
                'num_res_blocks': 2,
                'resblock_updown': True,
                'use_fp16': device.type != 'cpu',
                'use_scale_shift_norm': True,
            }
        )
    elif os.path.isfile(diffusion_model):
        model_config.update(
            {
                'attention_resolutions': '16',
                'class_cond': False,
                'diffusion_steps': 1000,
                'rescale_timesteps': True,
                'timestep_respacing': 'ddim100',
                'image_size': 256,
                'learn_sigma': True,
                'noise_schedule': 'linear',
                'num_channels': 128,
                'num_heads': 1,
                'num_res_blocks': 2,
                'use_checkpoint': True,
                'use_fp16': device.type != 'cpu',
                'use_scale_shift_norm': False,
            }
        )

    secondary_model = None
    if use_secondary_model:
        from discoart.nn.sec_diff import SecondaryDiffusionImageNet2

        secondary_model = SecondaryDiffusionImageNet2()
        secondary_model.load_state_dict(
            torch.load(
                f'{cache_dir}/secondary_model_imagenet_2.pth', map_location='cpu'
            )
        )
        secondary_model.eval().requires_grad_(False).to(device)

    return model_config, secondary_model


def load_diffusion_model(model_config, diffusion_model, steps, device):
    from guided_diffusion.script_util import (
        create_model_and_diffusion,
    )

    timestep_respacing = f'ddim{steps}'
    diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
    model_config.update(
        {
            'timestep_respacing': timestep_respacing,
            'diffusion_steps': diffusion_steps,
        }
    )

    model, diffusion = create_model_and_diffusion(**model_config)
    if os.path.isfile(diffusion_model):
        logger.debug(f'loading customized diffusion model from {diffusion_model}')
        _model_path = diffusion_model
    else:
        _model_path = f'{cache_dir}/{diffusion_model}.pt'
    model.load_state_dict(torch.load(_model_path, map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()

    return model, diffusion


class PromptParser(SimpleTokenizer):
    def __init__(self, on_misspelled_token: str, **kwargs):
        super().__init__(**kwargs)
        self.spell = SpellChecker()
        from . import __resources_path__

        with open(f'{__resources_path__}/vocab.txt') as fp:
            self.spell.word_frequency.load_words(
                line.strip() for line in fp if len(line.strip()) > 1
            )
        self.on_misspelled_token = on_misspelled_token

    @staticmethod
    def _split_weight(prompt):
        if ':' in prompt:
            vals = prompt.rsplit(':', 1)
        else:
            vals = [prompt, 1]
        return vals[0], float(vals[1])

    def parse(self, text: str) -> Tuple[str, float]:
        text, weight = self._split_weight(text)
        text = whitespace_clean(basic_clean(text)).lower()
        all_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            all_tokens.append(token)
        unknowns = [
            v
            for v in self.spell.unknown(all_tokens)
            if len(v) > 2 and self.spell.correction(v) != v
        ]
        if unknowns:
            pairs = []
            for v in unknowns:
                vc = self.spell.correction(v)
                pairs.append((v, vc))
                if self.on_misspelled_token == 'correct':
                    for idx, ov in enumerate(all_tokens):
                        if ov == v:
                            all_tokens[idx] = vc

            if pairs:
                warning_str = '\n'.join(
                    f'Misspelled `{v}`, do you mean `{vc}`?' for v, vc in pairs
                )
                if self.on_misspelled_token == 'raise':
                    raise ValueError(warning_str)
                elif self.on_misspelled_token == 'correct':
                    logger.warning(
                        'auto-corrected the following tokens:\n' + warning_str
                    )
                else:
                    logger.warning(
                        'Found misspelled tokens in the prompt:\n' + warning_str
                    )

        logger.debug(f'prompt: {all_tokens}, weight: {weight}')
        return ' '.join(all_tokens), weight
