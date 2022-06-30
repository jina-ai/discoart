import hashlib
import logging
import os
import subprocess
import sys
from os.path import expanduser
from pathlib import Path

import torch

cache_dir = f'{expanduser("~")}/.cache/{__package__}'


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
    logger.addHandler(ch)
    return logger


logger = _get_logger()

if not os.path.exists(cache_dir):
    logger.info(
        f'looks like you are running {__package__} for the first time, this will take some time. '
        f'You wont see this message on the second run.'
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

logger.debug(f'`.cache` dir is set to: {cache_dir}')

check_model_SHA = False


def _gitclone(url, dest):
    res = subprocess.run(
        ['git', 'clone', '--depth', '1', url, dest], stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    logger.debug(f'cloned {url} to {dest}: {res}')


def _clone_repo_install(repo_url, repo_dir):
    if not os.path.exists(repo_dir):
        _gitclone(repo_url, repo_dir)
    sys.path.append(repo_dir)


def _clone_dependencies():
    _clone_repo_install('https://github.com/openai/CLIP', f'{cache_dir}/CLIP')
    _clone_repo_install(
        'https://github.com/crowsonkb/guided-diffusion', f'{cache_dir}/guided_diffusion'
    )
    _clone_repo_install(
        'https://github.com/assafshocher/ResizeRight', f'{cache_dir}/resize_right'
    )


def _wget(url, outputdir):
    res = subprocess.run(
        ['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    logger.debug(res)


def _load_clip_model(device):
    from CLIP import clip

    clip_models = []
    ViTB32 = True  # @param{type:"boolean"}
    ViTB16 = True  # @param{type:"boolean"}
    ViTL14 = False  # @param{type:"boolean"}
    RN101 = False  # @param{type:"boolean"}
    RN50 = True  # @param{type:"boolean"}
    RN50x4 = False  # @param{type:"boolean"}
    RN50x16 = False  # @param{type:"boolean"}
    RN50x64 = False  # @param{type:"boolean"}
    if ViTB32 is True:
        clip_models.append(
            clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if ViTB16 is True:
        clip_models.append(
            clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if ViTL14 is True:
        clip_models.append(
            clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if RN50 is True:
        clip_models.append(
            clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if RN50x4 is True:
        clip_models.append(
            clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if RN50x16 is True:
        clip_models.append(
            clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if RN50x64 is True:
        clip_models.append(
            clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if RN101 is True:
        clip_models.append(
            clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(device)
        )
    return clip_models


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
                    load_all_models(diffusion_model, use_secondary_model, True)
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
                load_all_models(diffusion_model, True)
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
                    load_all_models(diffusion_model, use_secondary_model, True)
            else:
                logger.debug("512 Model SHA doesn't match, redownloading...")
                _wget(model_512_link, cache_dir)
                if os.path.exists(model_512_path):
                    model_512_downloaded = True
                else:
                    logger.debug('First URL Failed using FallBack')
                    load_all_models(diffusion_model, use_secondary_model, True)
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
                    load_all_models(diffusion_model, use_secondary_model, True)
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
                load_all_models(diffusion_model, use_secondary_model, True)

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
                'use_fp16': device != 'cpu',
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
                'use_fp16': device != 'cpu',
                'use_scale_shift_norm': True,
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

    return model_config, _load_clip_model(device), secondary_model


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
    model.load_state_dict(
        torch.load(f'{cache_dir}/{diffusion_model}.pt', map_location='cpu')
    )
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()

    return model, diffusion


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals) :]
    return vals[0], float(vals[1])
