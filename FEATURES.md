# DiscoArt vs. DD5.5

DiscoArt and DD5.5 follow the same algorithms, but DiscoArt has a much better  implementation ([guaranteed by me](https://github.com/hanxiao)) with **top** code quality, **the best** user experience and **rich** features. It is perfect for integration to professional developers.

DiscoArt is synced with the upstream update of original DD notebook. The list below summarizes the major differences between DiscoArt and DD5.5:

## Refactor & bug fixes
- Completely refactored the notebook implementation and aim for *best-in-class* quality. (I'm serious about this.)
- Fixes multiple bugs e.g. weighted prompts, cut scheduling in original DD5.5, which improves the generation quality.
- No dependency of IPython when not using in notebook/colab.
- Robust persistent storage for the generated images.
- Simpler interface and Pythonic API.

## Scheduling
- `cut_schedules_group` allows one to load in multiple cut schedules at once.
- `cut_ic_pow` is changed to a scheduling parameter to control the power of inner cut, the syntax is the same as `cut_overview`, `cut_innercut`.
- `clip_models_schedules` is added to control the scheduling of clip models, the syntax is the same as `cut_overview` but as a bool list `[True]*400+[False]*600`.
- `use_secondary_model` supports scheduling expressions, e.g. `[True]*400+[False]*600`.

## Customized diffusion
- Support default 512x512, 256x256 diffusion model as well as Pixel Art Diffusion, Watercolor Diffusion, and Pulp SciFi Diffusion models.
- `diffusion_model` and `diffusion_model_config` can be specified load custom diffusion model and override the default diffusion model.

## Feature changes
- DiscoArt does not support video generation and `image_prompt` (which was marked as ineffective in DD 5.4).
- Due to no video support, `text_prompts` in DiscoArt accepts a string or a list of strings, not a dictionary; i.e. no frame index `0:` or `100:`.
- `clip_models` accepts a list of values from [all open-clip pretrained models and weights](https://github.com/jina-ai/discoart/blob/main/discoart/resources/docstrings.yml#L90).
