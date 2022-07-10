import os
import warnings
from types import SimpleNamespace

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


__version__ = '0.1.7'


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

from .helper import load_all_models, load_diffusion_model, load_clip_models

model_config, secondary_model = load_all_models(
    '512x512_diffusion_uncond_finetune_008100',
    use_secondary_model=True,
    device=device,
)

from typing import TYPE_CHECKING, overload, List, Optional

if TYPE_CHECKING:
    from docarray import DocumentArray, Document

_clip_models_cache = {}


# begin_create_overload


@overload
def create(
    text_prompts: Optional[List[str]] = [
        'A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.',
        'yellow color scheme',
    ],
    init_image: Optional[str] = None,
    width_height: Optional[List[int]] = [1280, 768],
    skip_steps: Optional[int] = 0,
    steps: Optional[int] = 250,
    init_scale: Optional[int] = 1000,
    clip_guidance_scale: Optional[int] = 5000,
    tv_scale: Optional[int] = 0,
    range_scale: Optional[int] = 150,
    sat_scale: Optional[int] = 0,
    cutn_batches: Optional[int] = 4,
    diffusion_model: Optional[str] = '512x512_diffusion_uncond_finetune_008100',
    use_secondary_model: Optional[bool] = True,
    diffusion_sampling_mode: Optional[str] = 'ddim',
    perlin_init: Optional[bool] = False,
    perlin_mode: Optional[str] = 'mixed',
    seed: Optional[int] = None,
    eta: Optional[float] = 0.8,
    clamp_grad: Optional[bool] = True,
    clamp_max: Optional[float] = 0.05,
    randomize_class: Optional[bool] = True,
    clip_denoised: Optional[bool] = False,
    fuzzy_prompt: Optional[bool] = False,
    rand_mag: Optional[float] = 0.05,
    cut_overview: Optional[str] = '[12]*400+[4]*600',
    cut_innercut: Optional[str] = '[4]*400+[12]*600',
    cut_icgray_p: Optional[str] = '[0.2]*400+[0]*600',
    cut_ic_pow: Optional[str] = '[1]*1000',
    display_rate: Optional[int] = 20,
    n_batches: Optional[int] = 4,
    batch_size: Optional[int] = 1,
    batch_name: Optional[str] = '',
    clip_models: Optional[List[str]] = [
        'ViT-B-32::openai',
        'ViT-B-16::openai',
        'RN50::openai',
    ],
    use_vertical_symmetry: Optional[bool] = False,
    use_horizontal_symmetry: Optional[bool] = False,
    transformation_percent: Optional[List[float]] = [0.09],
    skip_augs: Optional[bool] = False,
    on_misspelled_token: Optional[str] = 'ignore',
) -> Optional['DocumentArray']:
    ...


# end_create_overload


@overload
def create(init_document: 'Document') -> Optional['DocumentArray']:
    ...


def create(**kwargs) -> Optional['DocumentArray']:
    """
    Create Disco Diffusion artworks and save the result into a DocumentArray.
    :param init_document: its ``.tags`` will be used as parameters, ``.uri`` (if present) will be used as init image.
    :param text_prompts: Phrase, sentence, or string of words and phrases describing what the image should look like.  The words will be analyzed by the AI and will guide the diffusion process toward the image(s) you describe. These can include commas and weights to adjust the relative importance of each element.  E.g. "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."Notice that this prompt loosely follows a structure: [subject], [prepositional details], [setting], [meta modifiers and artist]; this is a good starting point for your experiments. Developing text prompts takes practice and experience, and is not the subject of this guide.  If you are a beginner to writing text prompts, a good place to start is on a simple AI art app like Night Cafe, starry ai or WOMBO prior to using DD, to get a feel for how text gets translated into images by GAN tools.  These other apps use different technologies, but many of the same principles apply.
    :param init_image: Recall that in the image sequence above, the first image shown is just noise.  If an init_image is provided, diffusion will replace the noise with the init_image as its starting state.  To use an init_image, upload the image to the Colab instance or your Google Drive, and enter the full image path here. If using an init_image, you may need to increase skip_steps to ~ 50% of total steps to retain the character of the init. See skip_steps above for further discussion.
    :param width_height: Desired final image size, in pixels. You can have a square, wide, or tall image, but each edge length should be set to a multiple of 64px, and a minimum of 512px on the default CLIP model setting.  If you forget to use multiples of 64px in your dimensions, DD will adjust the dimensions of your image to make it so.
    :param skip_steps: Consider the chart shown here.  Noise scheduling (denoise strength) starts very high and progressively gets lower and lower as diffusion steps progress. The noise levels in the first few steps are very high, so images change dramatically in early steps.As DD moves along the curve, noise levels (and thus the amount an image changes per step) declines, and image coherence from one step to the next increases.The first few steps of denoising are often so dramatic that some steps (maybe 10-15% of total) can be skipped without affecting the final image. You can experiment with this as a way to cut render times.If you skip too many steps, however, the remaining noise may not be high enough to generate new content, and thus may not have ‘time left’ to finish an image satisfactorily.Also, depending on your other settings, you may need to skip steps to prevent CLIP from overshooting your goal, resulting in ‘blown out’ colors (hyper saturated, solid white, or solid black regions) or otherwise poor image quality.  Consider that the denoising process is at its strongest in the early steps, so skipping steps can sometimes mitigate other problems.Lastly, if using an init_image, you will need to skip ~50% of the diffusion steps to retain the shapes in the original init image. However, if you’re using an init_image, you can also adjust skip_steps up or down for creative reasons.  With low skip_steps you can get a result "inspired by" the init_image which will retain the colors and rough layout and shapes but look quite different. With high skip_steps you can preserve most of the init_image contents and just do fine tuning of the texture.
    :param steps: When creating an image, the denoising curve is subdivided into steps for processing. Each step (or iteration) involves the AI looking at subsets of the image called ‘cuts’ and calculating the ‘direction’ the image should be guided to be more like the prompt. Then it adjusts the image with the help of the diffusion denoiser, and moves to the next step.Increasing steps will provide more opportunities for the AI to adjust the image, and each adjustment will be smaller, and thus will yield a more precise, detailed image.  Increasing steps comes at the expense of longer render times.  Also, while increasing steps should generally increase image quality, there is a diminishing return on additional steps beyond 250 - 500 steps.  However, some intricate images can take 1000, 2000, or more steps.  It is really up to the user.  Just know that the render time is directly related to the number of steps, and many other parameters have a major impact on image quality, without costing additional time.
    :param cut_ic_pow: This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall image coherency and/or it may cause an undesirable ‘mosaic’ effect.   Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping with some details.
    :param init_scale: This controls how strongly CLIP will try to match the init_image provided.  This is balanced against the clip_guidance_scale (CGS) above.  Too much init scale, and the image won’t change much during diffusion. Too much CGS and the init image will be lost.
    :param clip_guidance_scale: CGS is one of the most important parameters you will use. It tells DD how strongly you want CLIP to move toward your prompt each timestep.  Higher is generally better, but if CGS is too strong it will overshoot the goal and distort the image. So a happy medium is needed, and it takes experience to learn how to adjust CGS. Note that this parameter generally scales with image dimensions. In other words, if you increase your total dimensions by 50% (e.g. a change from 512 x 512 to 512 x 768), then to maintain the same effect on the image, you’d want to increase clip_guidance_scale from 5000 to 7500. Of the basic settings, clip_guidance_scale, steps and skip_steps are the most important contributors to image quality, so learn them well.
    :param tv_scale: Total variance denoising. Optional, set to zero to turn off. Controls ‘smoothness’ of final output. If used, tv_scale will try to smooth out your final image to reduce overall noise. If your image is too ‘crunchy’, increase tv_scale. TV denoising is good at preserving edges while smoothing away noise in flat regions.  See https://en.wikipedia.org/wiki/Total_variation_denoising
    :param range_scale: Optional, set to zero to turn off.  Used for adjustment of color contrast.  Lower range_scale will increase contrast. Very low numbers create a reduced color palette, resulting in more vibrant or poster-like images. Higher range_scale will reduce contrast, for more muted images.
    :param sat_scale: Saturation scale. Optional, set to zero to turn off.  If used, sat_scale will help mitigate oversaturation. If your image is too saturated, increase sat_scale to reduce the saturation.
    :param cutn_batches: Each iteration, the AI cuts the image into smaller pieces known as cuts, and compares each cut to the prompt to decide how to guide the next diffusion step.  More cuts can generally lead to better images, since DD has more chances to fine-tune the image precision in each timestep.  Additional cuts are memory intensive, however, and if DD tries to evaluate too many cuts at once, it can run out of memory.  You can use cutn_batches to increase cuts per timestep without increasing memory usage. At the default settings, DD is scheduled to do 16 cuts per timestep.  If cutn_batches is set to 1, there will indeed only be 16 cuts total per timestep. However, if cutn_batches is increased to 4, DD will do 64 cuts total in each timestep, divided into 4 sequential batches of 16 cuts each.  Because the cuts are being evaluated only 16 at a time, DD uses the memory required for only 16 cuts, but gives you the quality benefit of 64 cuts.  The tradeoff, of course, is that this will take ~4 times as long to render each image.So, (scheduled cuts) x (cutn_batches) = (total cuts per timestep). Increasing cutn_batches will increase render times, however, as the work is being done sequentially.  DD’s default cut schedule is a good place to start, but the cut schedule can be adjusted in the Cutn Scheduling section, explained below.
    :param diffusion_model: Diffusion_model of choice.
    :param use_secondary_model: Option to use a secondary purpose-made diffusion model to clean up interim diffusion images for CLIP evaluation.    If this option is turned off, DD will use the regular (large) diffusion model.    Using the secondary model is faster - one user reported a 50% improvement in render speed! However, the secondary model is much smaller, and may reduce image quality and detail.  I suggest you experiment with this.
    :param diffusion_sampling_mode: Two alternate diffusion denoising algorithms. ddim has been around longer, and is more established and tested.  plms is a newly added alternate method that promises good diffusion results in fewer steps, but has not been as fully tested and may have side effects. This new plms mode is actively being researched in the #settings-and-techniques channel in the DD Discord.
    :param perlin_init: Normally, DD will use an image filled with random noise as a starting point for the diffusion curve.  If perlin_init is selected, DD will instead use a Perlin noise model as an initial state.  Perlin has very interesting characteristics, distinct from random noise, so it’s worth experimenting with this for your projects. Beyond perlin, you can, of course, generate your own noise images (such as with GIMP, etc) and use them as an init_image (without skipping steps). Choosing perlin_init does not affect the actual diffusion process, just the starting point for the diffusion. Please note that selecting a perlin_init will replace and override any init_image you may have specified.  Further, because the 2D, 3D and video animation systems all rely on the init_image system, if you enable Perlin while using animation modes, the perlin_init will jump in front of any previous image or video input, and DD will NOT give you the expected sequence of coherent images. All of that said, using Perlin and animation modes together do make a very colorful rainbow effect, which can be used creatively.
    :param perlin_mode: sets type of Perlin noise: colored, gray, or a mix of both, giving you additional options for noise types. Experiment to see what these do in your projects.
    :param seed: Deep in the diffusion code, there is a random number ‘seed’ which is used as the basis for determining the initial state of the diffusion.  By default, this is random, but you can also specify your own seed.  This is useful if you like a particular result and would like to run more iterations that will be similar. After each run, the actual seed value used will be reported in the parameters report, and can be reused if desired by entering seed # here.  If a specific numerical seed is used repeatedly, the resulting images will be quite similar but not identical.
    :param eta: eta (greek letter η) is a diffusion model variable that mixes in a random amount of scaled noise into each timestep. 0 is no noise, 1.0 is more noise. As with most DD parameters, you can go below zero for eta, but it may give you unpredictable results. The steps parameter has a close relationship with the eta parameter. If you set eta to 0, then you can get decent output with only 50-75 steps. Setting eta to 1.0 favors higher step counts, ideally around 250 and up. eta has a subtle, unpredictable effect on image, so you’ll need to experiment to see how this affects your projects.
    :param clamp_grad: As I understand it, clamp_grad is an internal limiter that stops DD from producing extreme results.  Try your images with and without clamp_grad. If the image changes drastically with clamp_grad turned off, it probably means your clip_guidance_scale is too high and should be reduced.
    :param clamp_max: Sets the value of the clamp_grad limitation. Default is 0.05, providing for smoother, more muted coloration in images, but setting higher values (0.15-0.3) can provide interesting contrast and vibrancy.
    :param fuzzy_prompt: Controls whether to add multiple noisy prompts to the prompt losses. If True, can increase variability of image output. Experiment with this.
    :param rand_mag: Affects only the fuzzy_prompt.  Controls the magnitude of the random noise added by fuzzy_prompt.
    :param cut_overview: The schedule of overview cuts
    :param cut_innercut: The schedule of inner cuts
    :param cut_icgray_p: This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall image coherency and/or it may cause an undesirable ‘mosaic’ effect.   Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping with some details.
    :param display_rate: During a diffusion run, you can monitor the progress of each image being created with this variable.  If display_rate is set to 50, DD will show you the in-progress image every 50 timesteps. Setting this to a lower value, like 5 or 10, is a good way to get an early peek at where your image is heading. If you don’t like the progression, just interrupt execution, change some settings, and re-run.  If you are planning a long, unmonitored batch, it’s better to set display_rate equal to steps, because displaying interim images does slow Colab down slightly.
    :param n_batches: This variable sets the number of still images you want DD to create.  If you are using an animation mode (see below for details) DD will ignore n_batches and create a single set of animated frames based on the animation settings.
    :param batch_name: The name of the batch, the batch id will be named as "discoart-[batch_name]-seed". To avoid your artworks be overridden by other users, please use a unique name.
    :param clip_models: CLIP Model selectors. These various CLIP models are available for you to use during image generation.  Models have different styles or ‘flavors,’ so look around.  You can mix in multiple models as well for different results. However, keep in mind that some models are extremely memory-hungry, and turning on additional models will take additional memory and may cause a crash.Also supported open_clip pretrained models, use `::` to separate model name and pretrained weight name, e.g. `ViT-B/32::laion2b_e16`. Full list of models and weights can be found here: https://github.com/mlfoundations/open_clip#pretrained-model-interface RN50::openai RN50::yfcc15m RN50::cc12m RN50-quickgelu::openai RN50-quickgelu::yfcc15m RN50-quickgelu::cc12m RN101::openai RN101::yfcc15m RN101-quickgelu::openai RN101-quickgelu::yfcc15m RN50x4::openai RN50x16::openai RN50x64::openai ViT-B-32::openai ViT-B-32::laion2b_e16 ViT-B-32::laion400m_e31 ViT-B-32::laion400m_e32 ViT-B-32-quickgelu::openai ViT-B-32-quickgelu::laion400m_e31 ViT-B-32-quickgelu::laion400m_e32 ViT-B-16::openai ViT-B-16::laion400m_e31 ViT-B-16::laion400m_e32 ViT-B-16-plus-240::laion400m_e31 ViT-B-16-plus-240::laion400m_e32 ViT-L-14::openai ViT-L-14-336::openai
    :param use_vertical_symmetry: Enforce symmetry over x axis of the image on [tr_ststeps for tr_st in transformation_steps] steps of the diffusion process
    :param use_horizontal_symmetry: Enforce symmetry over y axis of the image on [tr_ststeps for tr_st in transformation_steps] steps of the diffusion process
    :param transformation_percent: Steps expressed in percentages in which the symmetry is enforced
    :param skip_augs: Controls whether to skip torchvision augmentations
    :param on_misspelled_token: Strategy when encounter misspelled token, can be 'raise', 'correct' and 'ignore'. If 'raise', then the misspelled token in the prompt will raise a ValueError. If 'correct', then the token will be replaced with the correct token. If 'ignore', then the token will be ignored but a warning will show.
    :return: a DocumentArray object that has `n_batches` Documents
    """
    from .config import load_config, print_args_table, save_config_svg
    from .runner import do_run

    if 'init_document' in kwargs:
        d = kwargs['init_document']
        _kwargs = d.tags
        if not _kwargs:
            warnings.warn('init_document has no .tags, fallback to default config')
        if d.uri:
            _kwargs['init_image'] = kwargs['init_document'].uri
        else:
            warnings.warn('init_document has no .uri, fallback to no init image')
        kwargs.pop('init_document')
        if kwargs:
            warnings.warn(
                'init_document has .tags and .uri, but kwargs are also present, will override .tags'
            )
            _kwargs.update(kwargs)
        _args = load_config(user_config=_kwargs)
    else:
        _args = load_config(user_config=kwargs)

    save_config_svg(_args)

    _args = SimpleNamespace(**_args)

    model, diffusion = load_diffusion_model(
        model_config, _args.diffusion_model, steps=_args.steps, device=device
    )

    clip_models = load_clip_models(
        device, enabled=_args.clip_models, clip_models=_clip_models_cache
    )

    gc.collect()
    torch.cuda.empty_cache()

    try:
        do_run(_args, (model, diffusion, clip_models, secondary_model), device)
    except KeyboardInterrupt:
        pass
    except Exception as ex:
        from .helper import logger

        logger.error(f'{ex!r}')
    finally:

        _name = _args.name_docarray

        if not os.path.exists(f'{_name}.protobuf.lz4'):
            # not even a single document was created
            return

        from IPython import display

        display.clear_output(wait=True)

        from docarray import DocumentArray

        _da = DocumentArray.load_binary(f'{_name}.protobuf.lz4')
        if _da and _da[0].uri:
            _da.plot_image_sprites(
                skip_empty=True, show_index=True, keep_aspect_ratio=True
            )
        result = _da

        print_args_table(vars(_args))
        from IPython.display import FileLink, display

        persist_file = FileLink(
            f'{_name}.protobuf.lz4',
            result_html_prefix=f'▶ Download the local backup (in case cloud storage failed): ',
        )
        config_file = FileLink(
            f'{_name}.svg',
            result_html_prefix=f'▶ Download the config as SVG image: ',
        )
        display(config_file, persist_file)

        from rich import print
        from rich.markdown import Markdown

        md = Markdown(
            f'''
Results are stored in a [DocumentArray](https://docarray.jina.ai/fundamentals/documentarray/) and synced to the cloud.

You can simply pull it from any machine:

```python
# pip install docarray[common]
from docarray import DocumentArray

da = DocumentArray.pull('{_name}')
```

If for some reason the cloud storage is not available, you may also download the file manually and load it from local disk:

```python
da = DocumentArray.load_binary('{_name}.protobuf.lz4')
```

More usage such as plotting, post-analysis can be found in the [README](https://github.com/jina-ai/discoart).
        ''',
            code_theme='igor',
        )
        print(md)

        gc.collect()
        torch.cuda.empty_cache()

    return result
