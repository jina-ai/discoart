import copy
import multiprocessing
import os
import warnings
from types import SimpleNamespace
from typing import overload, List, Optional, Dict, Any, Union, TYPE_CHECKING


if TYPE_CHECKING:
    import threading
    import asyncio

from docarray import DocumentArray, Document

_clip_models_cache = {}


# begin_create_overload
@overload
def create(
    batch_name: Optional[str] = None,
    batch_size: Optional[int] = 1,
    clamp_grad: Optional[Union[bool, str]] = True,
    clamp_max: Optional[Union[float, str]] = 0.05,
    clip_denoised: Optional[bool] = False,
    clip_guidance_scale: Optional[Union[int, str]] = 5000,
    clip_models: Optional[List[str]] = [
        'ViT-B-32::openai',
        'ViT-B-16::openai',
        'RN50::openai',
    ],
    clip_models_schedules: Optional[Dict[str, Union[str, List[str]]]] = None,
    cut_ic_pow: Optional[Union[float, str]] = 1.0,
    cut_icgray_p: Optional[Union[float, str]] = '[0.2]*400+[0]*600',
    cut_innercut: Optional[Union[float, str]] = '[4]*400+[12]*600',
    cut_overview: Optional[Union[float, str]] = '[12]*400+[4]*600',
    cut_schedules_group: Optional[str] = None,
    cutn_batches: Optional[Union[int, str]] = 4,
    diffusion_model: Optional[str] = '512x512_diffusion_uncond_finetune_008100',
    diffusion_model_config: Optional[Dict[str, Any]] = None,
    diffusion_sampling_mode: Optional[str] = 'ddim',
    eta: Optional[float] = 0.8,
    gif_fps: Optional[int] = 20,
    gif_size_ratio: Optional[float] = 0.5,
    image_output: Optional[bool] = True,
    init_document: Optional[Union['Document', 'DocumentArray']] = None,
    init_image: Optional[str] = None,
    init_scale: Optional[Union[int, str]] = 1000,
    inpaint_image: Optional[str] = None,
    inpaint_mask: Optional[str] = None,
    n_batches: Optional[int] = 4,
    name_docarray: Optional[str] = None,
    on_misspelled_token: Optional[str] = 'ignore',
    perlin_init: Optional[bool] = False,
    perlin_mode: Optional[str] = 'mixed',
    rand_mag: Optional[float] = 0.05,
    randomize_class: Optional[bool] = True,
    range_scale: Optional[Union[int, str]] = 150,
    sat_scale: Optional[Union[int, str]] = 0,
    save_rate: Optional[int] = 20,
    seed: Optional[int] = None,
    skip_augs: Optional[Union[bool, str]] = False,
    skip_event: Optional[
        Union['multiprocessing.Event', 'asyncio.Event', 'threading.Event']
    ] = None,
    skip_steps: Optional[int] = 0,
    steps: Optional[int] = 250,
    stop_event: Optional[
        Union['multiprocessing.Event', 'asyncio.Event', 'threading.Event']
    ] = None,
    text_clip_on_cpu: Optional[bool] = False,
    text_prompts: Optional[Union[List[str], Dict[str, Any]]] = [
        'A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.',
        'yellow color scheme',
    ],
    transformation_percent: Optional[List[float]] = [0.09],
    truncate_overlength_prompt: Optional[bool] = False,
    tv_scale: Optional[Union[int, str]] = 0,
    use_horizontal_symmetry: Optional[bool] = False,
    use_secondary_model: Optional[Union[bool, str]] = True,
    use_vertical_symmetry: Optional[bool] = False,
    width_height: Optional[List[int]] = [1280, 768],
) -> Optional['DocumentArray']:

    ...


# end_create_overload


@overload
def create(init_document: 'Document') -> Optional['DocumentArray']:
    ...


# begin_create_docstring
def create(**kwargs) -> Optional['DocumentArray']:
    """
    Create Disco Diffusion artworks and return the result as a DocumentArray object.

    :param batch_name: The name of the batch, the batch id will be named as "discoart-[batch_name]-[uuid]". To avoid your artworks be overridden by other users, please use a unique name.
    :param batch_size: [DiscoArt] The number of samples generated at each steps. Say `batch_size=3`, then you can generate three images in one run. Not only this is faster than three runs, but it leverages loss function better and potentially yields higher quality images.One can of course also do `n_batches=3` and `batch_size=1` to generate three images in one run. But using `batch_size=3` is marginally faster and yield higher quality images.High `batch_size` can lead to OOM.
    :param clamp_grad: As I understand it, clamp_grad is an internal limiter that stops DD from producing extreme results.  Try your images with and without clamp_grad. If the image changes drastically with clamp_grad turned off, it probably means your clip_guidance_scale is too high and should be reduced.[DiscoArt] Can be scheduled via syntax `[val1]*400+[val2]*600`.
    :param clamp_max: Sets the value of the clamp_grad limitation. Default is 0.05, providing for smoother, more muted coloration in images, but setting higher values (0.15-0.3) can provide interesting contrast and vibrancy.[DiscoArt] Can be scheduled via syntax `[val1]*400+[val2]*600`.
    :param clip_denoised: Determines whether CLIP discriminates a noisy or denoised image
    :param clip_guidance_scale: CGS is one of the most important parameters you will use. It tells DD how strongly you want CLIP to move toward your prompt each timestep.  Higher is generally better, but if CGS is too strong it will overshoot the goal and distort the image. So a happy medium is needed, and it takes experience to learn how to adjust CGS. Note that this parameter generally scales with image dimensions. In other words, if you increase your total dimensions by 50% (e.g. a change from 512 x 512 to 512 x 768), then to maintain the same effect on the image, you’d want to increase clip_guidance_scale from 5000 to 7500. Of the basic settings, clip_guidance_scale, steps and skip_steps are the most important contributors to image quality, so learn them well.
    :param clip_models: [DiscoArt] CLIP Model selectors provided by open-clip package. These various CLIP models are available for you to use during image generation.  Models have different styles or ‘flavors,’ so look around.  You can mix in multiple models as well for different results. However, keep in mind that some models are extremely memory-hungry, and turning on additional models will take additional memory and may cause a crash.Also supported open_clip pretrained models, use `::` to separate model name and pretrained weight name, e.g. `ViT-B/32::laion2b_e16`. Full list of models and weights can be found here: https://github.com/mlfoundations/open_clip#pretrained-model-interface RN50::openai RN50::yfcc15m RN50::cc12m RN50-quickgelu::openai RN50-quickgelu::yfcc15m RN50-quickgelu::cc12m RN101::openai RN101::yfcc15m RN101-quickgelu::openai RN101-quickgelu::yfcc15m RN50x4::openai RN50x16::openai RN50x64::openai ViT-B-32::openai ViT-B-32::laion2b_e16 ViT-B-32::laion400m_e31 ViT-B-32::laion400m_e32 ViT-B-32-quickgelu::openai ViT-B-32-quickgelu::laion400m_e31 ViT-B-32-quickgelu::laion400m_e32 ViT-B-16::openai ViT-B-16::laion400m_e31 ViT-B-16::laion400m_e32 ViT-B-16-plus-240::laion400m_e31 ViT-B-16-plus-240::laion400m_e32 ViT-L-14::openai ViT-L-14-336::openai
    :param clip_models_schedules: [DiscoArt] A dictionary of string to boolean list that represents on/off of CLIP models at each step. CLIP Model schedules use a similar mechanism to cut_overview and `cut_innercut`. For example, `{"RN101::openai": "[True]*400+[False]*600"}` schedules RN101 to run for the first 40% of steps and then is no longer used for the remaining steps. `[True]*1000` is equivalent to always on and is the default if this parameter is not set. Note, the model must be included in the `clip_models` otherwise this parameter is ignored.
    :param cut_ic_pow: This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall image coherency and/or it may cause an undesirable ‘mosaic’ effect.   Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping with some details.[DiscoArt] This can be a list of floats that represents the value at different steps, the syntax follows the same as `cut_overview`.
    :param cut_icgray_p: In addition to the overall cut schedule, a portion of the cuts can be set to be grayscale instead of color. This may help with improved definition of shapes and edges, especially in the early diffusion steps where the image structure is being defined.
    :param cut_innercut: The schedule of inner cuts, which are smaller cropped images from the interior of the image, helpful in tuning fine details. The size of the inner cuts can be adjusted using the `cut_ic_pow` parameter.
    :param cut_overview: The schedule of overview cuts, which take a snapshot of the entire image and evaluate that against the prompt.
    :param cut_schedules_group: [DiscoArt] The group name of the cut-scheduling. For example, `default` corresponds to default:  cut_overview: "[12]*400+[4]*600"  cut_innercut: "[4]*400+[12]*600"  cut_ic_pow: "[1]*1000"  cut_icgray_p: "[0.2]*400+[0]*600"There are 2 predefined groups: `pad_or_pulp` and `watercolor`.This parameter has less priority than the cut_overview, cut_innercut, cut_ic_pow, cut_icgray_p parameters, meaning that one can override cut_overview, cut_innercut, cut_ic_pow, cut_icgray_p by setting them afterwards.
    :param cutn_batches: Each iteration, the AI cuts the image into smaller pieces known as cuts, and compares each cut to the prompt to decide how to guide the next diffusion step.  More cuts can generally lead to better images, since DD has more chances to fine-tune the image precision in each timestep.  Additional cuts are memory intensive, however, and if DD tries to evaluate too many cuts at once, it can run out of memory.  You can use cutn_batches to increase cuts per timestep without increasing memory usage. At the default settings, DD is scheduled to do 16 cuts per timestep.  If cutn_batches is set to 1, there will indeed only be 16 cuts total per timestep. However, if cutn_batches is increased to 4, DD will do 64 cuts total in each timestep, divided into 4 sequential batches of 16 cuts each.  Because the cuts are being evaluated only 16 at a time, DD uses the memory required for only 16 cuts, but gives you the quality benefit of 64 cuts.  The tradeoff, of course, is that this will take ~4 times as long to render each image.So, (scheduled cuts) x (cutn_batches) = (total cuts per timestep). Increasing cutn_batches will increase render times, however, as the work is being done sequentially.  DD’s default cut schedule is a good place to start, but the cut schedule can be adjusted in the Cutn Scheduling section, explained below.[DiscoArt] Can be scheduled via syntax `[val1]*400+[val2]*600`.
    :param diffusion_model: Diffusion_model of choice. Note that you don't have to write the full name of the diffusion model, e.g. any prefix is enough.To use a listed all diffusion models, you can do:```pythonfrom discoart import createcreate(diffusion_model='portrait_generator', ...)```
    :param diffusion_model_config: [DiscoArt] The customized diffusion model config as a dictionary, if specified will override the values with the same name in the default model config.
    :param diffusion_sampling_mode: Two alternate diffusion denoising algorithms. ddim has been around longer, and is more established and tested.  plms is a newly added alternate method that promises good diffusion results in fewer steps, but has not been as fully tested and may have side effects. This new plms mode is actively being researched in the #settings-and-techniques channel in the DD Discord.
    :param eta: eta (greek letter η) is a diffusion model variable that mixes in a random amount of scaled noise into each timestep. 0 is no noise, 1.0 is more noise. As with most DD parameters, you can go below zero for eta, but it may give you unpredictable results. The steps parameter has a close relationship with the eta parameter. If you set eta to 0, then you can get decent output with only 50-75 steps. Setting eta to 1.0 favors higher step counts, ideally around 250 and up. eta has a subtle, unpredictable effect on image, so you’ll need to experiment to see how this affects your projects.
    :param gif_fps: [DiscoArt] The frame rate of the generated GIF. Set it to -1 for not saving GIF.
    :param gif_size_ratio: [DiscoArt] The relative size vs. the original image, small size ratio gives smaller file size.
    :param image_output: [DiscoArt] If set, then output will be saved as images. This includes intermediate, final results in the form of PNG and GIF. If set to False, then no images will be saved, everything will be saved in a Protobuf LZ4 format. https://docarray.jina.ai/fundamentals/documentarray/serialization/#from-to-bytes
    :param init_document: [DiscoArt] Use a Document object as the initial state for DD: its ``.tags`` will be used as parameters, ``.uri`` (if present) will be used as init image.
    :param init_image: Recall that in the image sequence above, the first image shown is just noise.  If an init_image is provided, diffusion will replace the noise with the init_image as its starting state.  To use an init_image, upload the image to the Colab instance or your Google Drive, and enter the full image path here. If using an init_image, you may need to increase skip_steps to ~ 50% of total steps to retain the character of the init. See skip_steps above for further discussion.
    :param init_scale: This controls how strongly CLIP will try to match the init_image provided.  This is balanced against the clip_guidance_scale (CGS) above.  Too much init scale, and the image won’t change much during diffusion. Too much CGS and the init image will be lost.[DiscoArt] Can be scheduled via syntax `[val1]*400+[val2]*600`.
    :param inpaint_image: [DiscoArt] If an inpaint image and mask is provided, final image will match the inpaint image in non-masked areas.
    :param inpaint_mask: [DiscoArt] If an inpaint image and mask is provided, final image will match the inpaint image in non-masked areas.
    :param n_batches: This variable sets the number of still images you want DD to create.  If you are using an animation mode (see below for details) DD will ignore n_batches and create a single set of animated frames based on the animation settings.
    :param name_docarray: [DiscoArt] When specified, it overrides the default naming schema of the resulted DocumentArray. Useful when you have to know the result DocumentArray name in advance.The name also supports variable substitution via `{}`. For example, `name_docarray='test-{steps}-{perlin_init}'` will give the name of the DocumentArray as `test-250-False`. Any variable in the config can be substituted.
    :param on_misspelled_token: [DiscoArt] Strategy when encounter misspelled token, can be 'raise', 'correct' and 'ignore'. If 'raise', then the misspelled token in the prompt will raise a ValueError. If 'correct', then the token will be replaced with the correct token. If 'ignore', then the token will be ignored but a warning will show.
    :param perlin_init: Normally, DD will use an image filled with random noise as a starting point for the diffusion curve.  If perlin_init is selected, DD will instead use a Perlin noise model as an initial state.  Perlin has very interesting characteristics, distinct from random noise, so it’s worth experimenting with this for your projects. Beyond perlin, you can, of course, generate your own noise images (such as with GIMP, etc) and use them as an init_image (without skipping steps). Choosing perlin_init does not affect the actual diffusion process, just the starting point for the diffusion. Please note that selecting a perlin_init will replace and override any init_image you may have specified.  Further, because the 2D, 3D and video animation systems all rely on the init_image system, if you enable Perlin while using animation modes, the perlin_init will jump in front of any previous image or video input, and DD will NOT give you the expected sequence of coherent images. All of that said, using Perlin and animation modes together do make a very colorful rainbow effect, which can be used creatively.
    :param perlin_mode: sets type of Perlin noise: colored, gray, or a mix of both, giving you additional options for noise types. Experiment to see what these do in your projects.
    :param rand_mag: Affects only the fuzzy_prompt.  Controls the magnitude of the random noise added by fuzzy_prompt.
    :param randomize_class: Controls whether the imagenet class is randomly changed each iteration
    :param range_scale: Optional, set to zero to turn off.  Used for adjustment of color contrast.  Lower range_scale will increase contrast. Very low numbers create a reduced color palette, resulting in more vibrant or poster-like images. Higher range_scale will reduce contrast, for more muted images.[DiscoArt] Can be scheduled via syntax `[val1]*400+[val2]*600`.
    :param sat_scale: Saturation scale. Optional, set to zero to turn off.  If used, sat_scale will help mitigate oversaturation. If your image is too saturated, increase sat_scale to reduce the saturation.[DiscoArt] Can be scheduled via syntax `[val1]*400+[val2]*600`.
    :param save_rate: [DiscoArt] The number of steps to save intermediate results. It is a replacement to original `display_rate` parameter. Set it to -1 for not saving any intermediate result.
    :param seed: Deep in the diffusion code, there is a random number ‘seed’ which is used as the basis for determining the initial state of the diffusion.  By default, this is random, but you can also specify your own seed.  This is useful if you like a particular result and would like to run more iterations that will be similar. After each run, the actual seed value used will be reported in the parameters report, and can be reused if desired by entering seed # here.  If a specific numerical seed is used repeatedly, the resulting images will be quite similar but not identical.
    :param skip_augs: Controls whether to skip torchvision augmentations.[DiscoArt] Can be scheduled via syntax `[val1]*400+[val2]*600`.
    :param skip_event: [DiscoArt] A multiprocessing/asyncio/threading.Event that once set, will skip the current run and move to the next run as defined in `n_batches`.
    :param skip_steps: Consider the chart shown here.  Noise scheduling (denoise strength) starts very high and progressively gets lower and lower as diffusion steps progress. The noise levels in the first few steps are very high, so images change dramatically in early steps.As DD moves along the curve, noise levels (and thus the amount an image changes per step) declines, and image coherence from one step to the next increases.The first few steps of denoising are often so dramatic that some steps (maybe 10-15% of total) can be skipped without affecting the final image. You can experiment with this as a way to cut render times.If you skip too many steps, however, the remaining noise may not be high enough to generate new content, and thus may not have ‘time left’ to finish an image satisfactorily.Also, depending on your other settings, you may need to skip steps to prevent CLIP from overshooting your goal, resulting in ‘blown out’ colors (hyper saturated, solid white, or solid black regions) or otherwise poor image quality.  Consider that the denoising process is at its strongest in the early steps, so skipping steps can sometimes mitigate other problems.Lastly, if using an init_image, you will need to skip ~50% of the diffusion steps to retain the shapes in the original init image. However, if you’re using an init_image, you can also adjust skip_steps up or down for creative reasons.  With low skip_steps you can get a result "inspired by" the init_image which will retain the colors and rough layout and shapes but look quite different. With high skip_steps you can preserve most of the init_image contents and just do fine tuning of the texture.
    :param steps: When creating an image, the denoising curve is subdivided into steps for processing. Each step (or iteration) involves the AI looking at subsets of the image called ‘cuts’ and calculating the ‘direction’ the image should be guided to be more like the prompt. Then it adjusts the image with the help of the diffusion denoiser, and moves to the next step.Increasing steps will provide more opportunities for the AI to adjust the image, and each adjustment will be smaller, and thus will yield a more precise, detailed image.  Increasing steps comes at the expense of longer render times.  Also, while increasing steps should generally increase image quality, there is a diminishing return on additional steps beyond 250 - 500 steps.  However, some intricate images can take 1000, 2000, or more steps.  It is really up to the user.  Just know that the render time is directly related to the number of steps, and many other parameters have a major impact on image quality, without costing additional time.
    :param stop_event: [DiscoArt] A multiprocessing/asyncio/threading.Event that once set, will stop all generation of `n_batches` and immediately return from `create`.
    :param text_clip_on_cpu: [DiscoArt] Place text transformers of CLIP models on CPU. This saves more VRAM and will not hurt the speed at all on T4, P100, 3090; however, there are few community members report issue on V100 when it is `False`.
    :param text_prompts: Phrase, sentence, or string of words and phrases describing what the image should look like.  The words will be analyzed by the AI and will guide the diffusion process toward the image(s) you describe. These can include commas and weights to adjust the relative importance of each element.  E.g. "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."Notice that this prompt loosely follows a structure: [subject], [prepositional details], [setting], [meta modifiers and artist]; this is a good starting point for your experiments. Developing text prompts takes practice and experience, and is not the subject of this guide.  If you are a beginner to writing text prompts, a good place to start is on a simple AI art app like Night Cafe, starry ai or WOMBO prior to using DD, to get a feel for how text gets translated into images by GAN tools.  These other apps use different technologies, but many of the same principles apply.You can add weight at the end of each prompt string, say `:10` for positive weights and `:-3` for negative weights. [DiscoArt] Unlike original DD notebook, `text_prompts` does not need to be indexed by the timestamp. It is a list of strings.
    :param transformation_percent: Steps expressed in percentages in which the symmetry is enforced
    :param truncate_overlength_prompt: [DiscoArt] all CLIP models use 77 as the context length. Set this parameter truncates the prompt to the length of the model's context length.
    :param tv_scale: Total variance denoising. Optional, set to zero to turn off. Controls ‘smoothness’ of final output. If used, tv_scale will try to smooth out your final image to reduce overall noise. If your image is too ‘crunchy’, increase tv_scale. TV denoising is good at preserving edges while smoothing away noise in flat regions.  See https://en.wikipedia.org/wiki/Total_variation_denoising[DiscoArt] Can be scheduled via syntax `[val1]*400+[val2]*600`.
    :param use_horizontal_symmetry: Enforce symmetry over y axis of the image on [tr_ststeps for tr_st in transformation_steps] steps of the diffusion process
    :param use_secondary_model: Option to use a secondary purpose-made diffusion model to clean up interim diffusion images for CLIP evaluation.    If this option is turned off, DD will use the regular (large) diffusion model.    Using the secondary model is faster - one user reported a 50% improvement in render speed! However, the secondary model is much smaller, and may reduce image quality and detail.  I suggest you experiment with this.[DiscoArt] It can be also an boolean list schedule that represents on/off on secondary model at each step, same as `clip_models_schedules` or `cut_overview`.Note that without secondary model it will consume higher VRAM but gives better quality. Simply put, secondary model is faster, but a less accurate approximation to `p_mean_variance`.
    :param use_vertical_symmetry: Enforce symmetry over x axis of the image on [tr_ststeps for tr_st in transformation_steps] steps of the diffusion process
    :param width_height: Desired final image size, in pixels. You can have a square, wide, or tall image, but each edge length should be set to a multiple of 64px, and a minimum of 512px on the default CLIP model setting.  If you forget to use multiples of 64px in your dimensions, DD will adjust the dimensions of your image to make it so.
    :return: a DocumentArray object that has `n_batches` Documents
    """
    # end_create_docstring

    events = tuple(
        kwargs.pop(v, multiprocessing.Event()) or multiprocessing.Event()
        for v in ('skip_event', 'stop_event')
    )

    from .config import load_config, print_args_table

    if 'init_document' in kwargs:
        d = kwargs['init_document']
        _kwargs = {}
        if d:
            if isinstance(d, str):
                d = DocumentArray.pull(d)[0]
            elif isinstance(d, DocumentArray):
                d = d[0]

            _kwargs = copy.deepcopy(d.tags)
            if not _kwargs:
                warnings.warn('init_document has no .tags, fallback to default config')
            if d.uri:
                _kwargs['init_image'] = d.uri
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

    print_args_table(_args)
    _args = SimpleNamespace(**_args)

    from .helper import (
        load_diffusion_model,
        load_clip_models,
        load_secondary_model,
        get_device,
        free_memory,
        show_result_summary,
        get_output_dir,
    )

    device = get_device()
    model, diffusion = load_diffusion_model(_args, device=device)

    clip_models = load_clip_models(
        device,
        enabled=_args.clip_models,
        clip_models=_clip_models_cache,
        text_clip_on_cpu=_args.text_clip_on_cpu,
    )
    secondary_model = load_secondary_model(_args, device=device)

    free_memory()
    is_exit0 = False
    try:
        from .runner import do_run

        da = do_run(
            _args,
            (model, diffusion, clip_models, secondary_model),
            device=device,
            events=events,
        )
        is_exit0 = True
        return da
    except KeyboardInterrupt:
        is_exit0 = True
    finally:
        free_memory()

        _name = _args.name_docarray

        pb_path = os.path.join(get_output_dir(_name), 'da.protobuf.lz4')

        if os.path.exists(pb_path) and is_exit0:
            _da = DocumentArray.load_binary(pb_path)

            if (
                'DISCOART_DISABLE_RESULT_SUMMARY' not in os.environ
                and 'DISCOART_DISABLE_IPYTHON' not in os.environ
            ):
                show_result_summary(_da, _name, _args)
