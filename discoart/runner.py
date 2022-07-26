import copy
import os
import random
import threading
from pathlib import Path

import clip
import lpips
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.nn.functional import normalize as normalize_fn
from docarray import DocumentArray, Document

from .config import print_args_table
from .helper import (
    logger,
    get_ipython_funcs,
    free_memory,
    _MAX_DIFFUSION_STEPS,
    _eval_scheduling_str,
    _get_current_schedule,
    _get_schedule_table,
)
from .nn.losses import spherical_dist_loss, tv_loss, range_loss
from .nn.make_cutouts import MakeCutoutsDango
from .nn.sec_diff import alpha_sigma_to_t
from .nn.transform import symmetry_transformation_fn
from .persist import _sample_thread, _persist_thread, _save_progress_thread
from .prompt import PromptPlanner


def do_run(args, models, device, events) -> 'DocumentArray':
    skip_event, stop_event = events

    output_dir = os.path.join(
        os.environ.get('DISCOART_OUTPUT_DIR', './'), args.name_docarray
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info('preparing models...')

    model, diffusion, clip_models, secondary_model = models
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    side_x, side_y = ((args.width_height[j] // 64) * 64 for j in (0, 1))

    schedule_table = _get_schedule_table(args)

    from .nn.perlin_noises import create_perlin_noise, regen_perlin

    skip_steps = args.skip_steps

    loss_values = []

    model_stats = []

    _dp1, _, _output_fn = get_ipython_funcs()
    _dp1.clear_output(wait=True)

    if isinstance(args.text_prompts, str):
        args.text_prompts = [args.text_prompts]

    prompts = PromptPlanner(args)

    text_device = torch.device('cpu') if args.text_clip_on_cpu else device

    for model_name, clip_model in clip_models.items():

        # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
        try:
            try:
                input_resolution = clip_model.visual.input_resolution
            except:
                input_resolution = clip_model.visual.image_size
            logger.debug(f'input_resolution of {model_name}: {input_resolution}')
        except:
            input_resolution = 224
            logger.debug(
                f'fail to find input_resolution for {model_name}, fall back to {input_resolution}'
            )

        schedules = [True] * _MAX_DIFFUSION_STEPS
        if args.clip_models_schedules and model_name in args.clip_models_schedules:
            schedules = _eval_scheduling_str(args.clip_models_schedules[model_name])

        clip_model_stats = {
            'model_name': model_name,
            'clip_model': clip_model,
            'prompt_embeds': [],
            'prompt_weights': [],
            'schedules': schedules,
            'input_resolution': input_resolution,
        }

        for _p in prompts:
            print(p)
            txt = clip_model.encode_text(
                clip.tokenize(_p.text, truncate=args.truncate_overlength_prompt).to(
                    text_device
                )
            )

            clip_model_stats['prompt_embeds'].append(txt)
            clip_model_stats['prompt_weights'].append(_p.weight)

        clip_model_stats['prompt_embeds'] = (
            torch.cat(clip_model_stats['prompt_embeds']).unsqueeze(0).to(device)
        )
        clip_model_stats['prompt_weights'] = torch.tensor(
            clip_model_stats['prompt_weights'], device=device
        )
        model_stats.append(clip_model_stats)

    init = None

    _set_seed(args.seed)
    if args.init_image:
        d = Document(uri=args.init_image).load_uri_to_image_tensor(side_x, side_y)
        init = TF.to_tensor(d.tensor).to(device).unsqueeze(0).mul(2).sub(1)

    if args.perlin_init:
        if args.perlin_mode == 'color':
            init = create_perlin_noise(
                [1.5**-i * 0.5 for i in range(12)],
                1,
                1,
                False,
                side_y,
                side_x,
                device,
            )
            init2 = create_perlin_noise(
                [1.5**-i * 0.5 for i in range(8)], 4, 4, False, side_y, side_x, device
            )
        elif args.perlin_mode == 'gray':
            init = create_perlin_noise(
                [1.5**-i * 0.5 for i in range(12)], 1, 1, True, side_y, side_x, device
            )
            init2 = create_perlin_noise(
                [1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_y, side_x, device
            )
        else:
            init = create_perlin_noise(
                [1.5**-i * 0.5 for i in range(12)],
                1,
                1,
                False,
                side_y,
                side_x,
                device,
            )
            init2 = create_perlin_noise(
                [1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_y, side_x, device
            )
        # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
        init = (
            TF.to_tensor(init)
            .add(TF.to_tensor(init2))
            .div(2)
            .to(device)
            .unsqueeze(0)
            .mul(2)
            .sub(1)
        )
        del init2

    cur_t = None

    def cond_fn(x, t, **kwargs):

        t_int = (
            int(t[0].item()) + 1
        )  # errors on last step without +1, need to find source

        num_step = _MAX_DIFFUSION_STEPS - t_int
        scheduler = _get_current_schedule(schedule_table, num_step)

        with torch.enable_grad():

            x = x.detach().requires_grad_()
            if scheduler.use_secondary_model:
                alpha = torch.tensor(
                    diffusion.sqrt_alphas_cumprod[cur_t],
                    device=device,
                    dtype=torch.float32,
                )
                sigma = torch.tensor(
                    diffusion.sqrt_one_minus_alphas_cumprod[cur_t],
                    device=device,
                    dtype=torch.float32,
                )
                cosine_t = alpha_sigma_to_t(alpha, sigma)
                out = secondary_model(x, cosine_t[None].repeat([x.shape[0]])).pred
            else:
                my_t = torch.ones([x.shape[0]], device=device, dtype=torch.long) * cur_t
                out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False)[
                    'pred_xstart'
                ]

            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out * fac + x * (1 - fac)

            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out)
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = (
                tv_losses.sum() * scheduler.tv_scale
                + range_losses.sum() * scheduler.range_scale
                + sat_losses.sum() * scheduler.sat_scale
            )
            if init is not None and scheduler.init_scale:
                init_losses = lpips_model(x_in, init)
                loss += init_losses.sum() * scheduler.init_scale

            x_in_grad = torch.autograd.grad(loss, x_in)[0]

            for model_stat in model_stats:

                if not model_stat['schedules'][num_step]:
                    continue

                active_prompt_ids = prompts.get_prompt_ids(
                    model_stat['model_name'], num_step
                )

                if active_prompt_ids:
                    masked_embeds = model_stat['prompt_embeds'][active_prompt_ids]
                    masked_weights = model_stat['prompt_weights'][active_prompt_ids]
                    if masked_weights.sum().abs() <= 1e-5:
                        logger.warning(
                            f'Zero sum weights for prompt ids: {active_prompt_ids}'
                        )
                    elif masked_weights.sum() < 0:
                        logger.warning(
                            f'Negative sum weights for prompt ids: {active_prompt_ids}'
                        )
                    masked_weights = normalize_fn(masked_weights)
                    logger.debug(f'Prompt weights: {masked_weights}')
                else:
                    continue

                for _ in range(scheduler.cutn_batches):
                    cuts = MakeCutoutsDango(
                        model_stat['input_resolution'],
                        Overview=scheduler.cut_overview,
                        InnerCrop=scheduler.cut_innercut,
                        IC_Size_Pow=scheduler.cut_ic_pow,
                        IC_Grey_P=scheduler.cut_icgray_p,
                        skip_augs=scheduler.skip_augs,
                    )
                    clip_in = normalize(cuts(x_in.add(1).div(2)))

                    image_embeds = (
                        model_stat['clip_model'].encode_image(clip_in).unsqueeze(1)
                    )

                    dists = spherical_dist_loss(
                        image_embeds,
                        masked_embeds,  # 1, 2, 512
                    )

                    dists = dists.view(
                        [
                            scheduler.cut_overview + scheduler.cut_innercut,
                            x.shape[0],
                            -1,
                        ]
                    )

                    cut_loss = dists.mul(masked_weights).sum(2).mean(0).sum()

                    x_in_grad += torch.autograd.grad(
                        cut_loss
                        * scheduler.clip_guidance_scale
                        / scheduler.cutn_batches,
                        x_in,
                    )[0]

        x_is_NaN = False
        if not torch.isnan(x_in_grad).any():
            grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
        else:
            x_is_NaN = True
            grad = torch.zeros_like(x)
            logger.warning(
                f'NaN detected in grad at the diffusion inner-step {num_step}, '
                f'if this message continues to show up, '
                f'then your image is not updated and further steps are unnecessary.'
            )

        loss_values.append(loss.detach().item())

        if scheduler.clamp_grad and not x_is_NaN:
            magnitude = grad.square().mean().sqrt()
            return (
                grad * magnitude.clamp(max=scheduler.clamp_max) / magnitude
            )  # min=-0.02, min=-clamp_max,
        return grad

    if args.diffusion_sampling_mode == 'ddim':
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    logger.info('creating artworks...')

    image_display = _output_fn()
    is_busy_evs = [threading.Event() for _ in range(3)]

    da_batches = DocumentArray()
    from rich.text import Text

    org_seed = args.seed
    for _nb in range(args.n_batches):

        # set seed for each image in the batch
        new_seed = org_seed + _nb
        _set_seed(new_seed)
        args.seed = new_seed
        pgbar = '▰' * (_nb + 1) + '▱' * (args.n_batches - _nb - 1)

        _dp1.display(
            Text(f'n_batches={args.n_batches}: {pgbar}'),
            print_args_table(vars(args), only_non_default=True, console_print=False),
            image_display,
        )
        free_memory()

        d = Document(tags=copy.deepcopy(vars(args)))
        _d_gif = Document()
        da_batches.append(d)

        cur_t = diffusion.num_timesteps - skip_steps - 1

        if args.perlin_init:
            init = regen_perlin(
                args.perlin_mode, side_y, side_x, device, args.batch_size
            )

        if args.diffusion_sampling_mode == 'ddim':
            samples = sample_fn(
                model,
                (args.batch_size, 3, side_y, side_x),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=args.randomize_class,
                eta=args.eta,
                transformation_fn=lambda x: symmetry_transformation_fn(
                    x, args.use_horizontal_symmetry, args.use_vertical_symmetry
                ),
                transformation_percent=args.transformation_percent,
            )
        else:
            samples = sample_fn(
                model,
                (args.batch_size, 3, side_y, side_x),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=args.randomize_class,
                order=2,
            )

        threads = []
        for j, sample in enumerate(samples):
            if skip_event.is_set() or stop_event.is_set():
                logger.debug('skip_event/stop_event is set, skipping this run')
                skip_event.clear()
                break

            cur_t -= 1

            is_save_step = j % (args.display_rate or args.save_rate) == 0 or cur_t == -1
            threads.append(
                _sample_thread(
                    sample,
                    _nb,
                    cur_t,
                    d,
                    _d_gif,
                    image_display,
                    j,
                    loss_values,
                    output_dir,
                    is_busy_evs[0],
                    is_save_step,
                )
            )

            if is_save_step:
                threads.append(
                    _save_progress_thread(
                        d, _d_gif, _nb, output_dir, args.gif_fps, args.gif_size_ratio
                    )
                )
                threads.extend(
                    _persist_thread(
                        da_batches,
                        args.name_docarray,
                        is_busy_evs[1:],
                        is_busy_evs[0],
                        is_completed=cur_t == -1,
                    )
                )

        for t in threads:
            t.join()
        _dp1.clear_output(wait=True)

        if stop_event.is_set():
            logger.debug('stop_event is set, skipping the while `n_batches`')
            stop_event.clear()
            break

    logger.info(f'done! {args.name_docarray}')

    return da_batches


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
