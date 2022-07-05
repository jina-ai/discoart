import gc
import random
from threading import Thread

import clip
import lpips
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from IPython import display
from docarray import DocumentArray, Document
from ipywidgets import Output

from .helper import parse_prompt, logger
from .nn.losses import spherical_dist_loss, tv_loss, range_loss
from .nn.make_cutouts import MakeCutoutsDango
from .nn.sec_diff import alpha_sigma_to_t


def do_run(args, models, device) -> 'DocumentArray':
    logger.info('preparing models...')
    model, diffusion, clip_models, secondary_model = models
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    side_x = (args.width_height[0] // 64) * 64
    side_y = (args.width_height[1] // 64) * 64
    cut_overview = eval(args.cut_overview)
    cut_innercut = eval(args.cut_innercut)
    cut_icgray_p = eval(args.cut_icgray_p)

    from .nn.perlin_noises import create_perlin_noise, regen_perlin

    seed = args.seed

    skip_steps = args.skip_steps

    loss_values = []

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    model_stats = []
    for clip_model in clip_models:
        model_stat = {
            'clip_model': None,
            'target_embeds': [],
            'make_cutouts': None,
            'weights': [],
        }
        model_stat['clip_model'] = clip_model

        if isinstance(args.text_prompts, str):
            args.text_prompts = [args.text_prompts]

        for prompt in args.text_prompts:
            txt, weight = parse_prompt(prompt)
            txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

            if args.fuzzy_prompt:
                for i in range(25):
                    model_stat['target_embeds'].append(
                        (txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(
                            0, 1
                        )
                    )
                    model_stat['weights'].append(weight)
            else:
                model_stat['target_embeds'].append(txt)
                model_stat['weights'].append(weight)

        model_stat['target_embeds'] = torch.cat(model_stat['target_embeds'])
        model_stat['weights'] = torch.tensor(model_stat['weights'], device=device)
        if model_stat['weights'].sum().abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        model_stat['weights'] /= model_stat['weights'].sum().abs()
        model_stats.append(model_stat)

    init = None
    if args.init_image:
        d = Document(uri=args.init_image).load_uri_to_image_tensor(side_x, side_y)
        init = TF.to_tensor(d.tensor).to(device).unsqueeze(0).mul(2).sub(1)

    if args.perlin_init:
        if args.perlin_mode == 'color':
            init = create_perlin_noise(
                [1.5 ** -i * 0.5 for i in range(12)], 1, 1, False
            )
            init2 = create_perlin_noise(
                [1.5 ** -i * 0.5 for i in range(8)], 4, 4, False
            )
        elif args.perlin_mode == 'gray':
            init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, True)
            init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True)
        else:
            init = create_perlin_noise(
                [1.5 ** -i * 0.5 for i in range(12)], 1, 1, False
            )
            init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True)
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

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x_is_NaN = False
            x = x.detach().requires_grad_()
            n = x.shape[0]
            if secondary_model:
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
                out = secondary_model(x, cosine_t[None].repeat([n])).pred
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
            else:
                my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                out = diffusion.p_mean_variance(
                    model, x, my_t, clip_denoised=False, model_kwargs={'y': y}
                )
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out['pred_xstart'] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
            for model_stat in model_stats:
                for i in range(args.cutn_batches):
                    t_int = (
                            int(t.item()) + 1
                    )  # errors on last step without +1, need to find source
                    # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                    try:
                        input_resolution = model_stat[
                            'clip_model'
                        ].visual.input_resolution
                    except:
                        input_resolution = 224

                    cuts = MakeCutoutsDango(
                        input_resolution,
                        Overview=cut_overview[1000 - t_int],
                        InnerCrop=cut_innercut[1000 - t_int],
                        IC_Size_Pow=args.cut_ic_pow,
                        IC_Grey_P=cut_icgray_p[1000 - t_int],
                    )
                    clip_in = normalize(cuts(x_in.add(1).div(2)))
                    image_embeds = (
                        model_stat['clip_model'].encode_image(clip_in).float()
                    )
                    dists = spherical_dist_loss(
                        image_embeds.unsqueeze(1),
                        model_stat['target_embeds'].unsqueeze(0),
                    )
                    dists = dists.view(
                        [
                            cut_overview[1000 - t_int] + cut_innercut[1000 - t_int],
                            n,
                            -1,
                        ]
                    )
                    losses = dists.mul(model_stat['weights']).sum(2).mean(0)
                    loss_values.append(
                        losses.sum().item()
                    )  # log loss, probably shouldn't do per cutn_batch
                    x_in_grad += (
                            torch.autograd.grad(
                                losses.sum() * args.clip_guidance_scale, x_in
                            )[0]
                            / args.cutn_batches
                    )
            tv_losses = tv_loss(x_in)
            if secondary_model:
                range_losses = range_loss(out)
            else:
                range_losses = range_loss(out['pred_xstart'])
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = (
                    tv_losses.sum() * args.tv_scale
                    + range_losses.sum() * args.range_scale
                    + sat_losses.sum() * args.sat_scale
            )
            if init is not None and args.init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * args.init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if not torch.isnan(x_in_grad).any():
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                x_is_NaN = True
                grad = torch.zeros_like(x)
        if args.clamp_grad and not x_is_NaN:
            magnitude = grad.square().mean().sqrt()
            return (
                    grad * magnitude.clamp(max=args.clamp_max) / magnitude
            )  # min=-0.02, min=-clamp_max,
        return grad

    if args.diffusion_sampling_mode == 'ddim':
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    logger.info('creating artwork...')

    image_display = Output()
    da_batches = DocumentArray()

    for _nb in range(args.n_batches):
        display.clear_output(wait=True)
        display.display(args.name_docarray, image_display)
        gc.collect()
        torch.cuda.empty_cache()

        d = Document(tags=vars(args))
        da_batches.append(d)

        cur_t = diffusion.num_timesteps - skip_steps - 1

        if args.perlin_init:
            init = regen_perlin(
                args.perlin_mode, args.side_y, side_x, device, args.batch_size
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
            cur_t -= 1
            with image_display:
                if j % args.display_rate == 0 or cur_t == -1:
                    for _, image in enumerate(sample['pred_xstart']):
                        image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                        image.save('progress.png')
                        c = Document(tags={'cur_t': cur_t})
                        c.load_pil_image_to_datauri(image)
                        d.chunks.append(c)
                        display.clear_output(wait=True)
                        display.display(display.Image('progress.png'))
                        d.chunks.plot_image_sprites(
                            f'{args.name_docarray}-progress-{_nb}.png',
                            skip_empty=True,
                            show_index=True,
                            keep_aspect_ratio=True,
                        )
                        _start_persist(threads, da_batches, args.name_docarray)

                    if cur_t == -1:
                        d.load_pil_image_to_datauri(image)

        for t in threads:
            t.join()
    display.clear_output(wait=True)
    logger.info(f'done! {args.name_docarray}')
    da_batches.plot_image_sprites(
        skip_empty=True, show_index=True, keep_aspect_ratio=True
    )
    from .config import load_config
    load_config(da_batches[0].tags)
    return da_batches


def _start_persist(threads, da_batches, name_docarray):
    t = Thread(
        target=_silent_push,
        args=(
            da_batches,
            name_docarray,
        ),
    )
    threads.append(t)
    t.start()


def _silent_push(da_batches: DocumentArray, name: str) -> None:
    try:
        da_batches.save_binary(name, protocol='protobuf', compress='lz4')
        da_batches.push(name)
    except Exception as ex:
        logger.debug(f'push failed: {ex}')
