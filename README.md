![](.github/banner.png)

<p align="center">
<b>Create compelling Disco Diffusion artworks in one line</b>
</p>

<p align=center>
<a href="https://pypi.org/project/discoart/"><img src="https://img.shields.io/pypi/v/discoart?style=flat-square&amp;label=Release" alt="PyPI"></a>
<a href="https://hub.docker.com/repository/docker/jinaai/discoart"><img alt="Docker Cloud Build Status" src="https://img.shields.io/docker/cloud/build/jinaai/discoart?logo=docker&logoColor=white&style=flat-square"></a>
<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-3.1k-blueviolet?logo=slack&amp;logoColor=white&style=flat-square"></a>
<a href="https://colab.research.google.com/github/jina-ai/discoart/blob/main/discoart.ipynb"><img src="https://img.shields.io/badge/Open-in%20Colab-brightgreen?logo=google-colab&style=flat-square" alt="Open in Google Colab"/></a>
</p>

DiscoArt is an elegant way of creating compelling Disco Diffusion<sup><a href="#example-application">[*]</a></sup> artworks for generative artists, AI enthusiasts and hard-core developers. DiscoArt has a modern & professional API with a beautiful codebase, ensuring high usability and maintainability. It introduces handy features such as result recovery and persistence, gRPC/HTTP serving w/o TLS, post-analysis, easing the integration to larger cross-modal or multi-modal applications.

<sub><sup><a id="example-application">[*]</a> 
Disco Diffusion is a Google Colab Notebook that leverages CLIP-Guided Diffusion to allow one to create compelling and beautiful images from text prompts.
</sup></sub>


💯 **Best-in-class**: top-notch code quality, correctness-first, minimum dependencies; including bug fixes, feature improvements vs. the original DD5.x. 

👼 **Available to all**: fully optimized for Google Colab *free tier*! Perfect for those who don't own GPU by themselves.

🎨 **Focus on create not code**: one-liner `create()` with a Pythonic interface, autocompletion in IDE, and powerful features. Fetch real-time results anywhere anytime, no more worry on session outrage on Google Colab. Set initial state easily for more efficient parameter exploration.

🏭 **Ready for integration & production**: built on top of [DocArray](https://github.com/jina-ai/docarray) data structure, enjoy smooth integration with [Jina](https://github.com/jina-ai/jina), [CLIP-as-service](https://github.com/jina-ai/clip-as-service) and other cross-/multi-modal applications.


## [Gallery with prompts](https://twitter.com/hxiao/status/1542967938369687552?s=20&t=DO27EKNMADzv4WjHLQiPFA) 
## Install

```bash
pip install discoart
```

- If you want to start a Jupyter Notebook on your own GPU machine, the easiest way is to [use our prebuilt Docker image](#run-in-docker).
- If you are not using Google Colab/Jupyter Notebook, then other dependencies might be required [as described in the Dockerfile](./Dockerfile).



## Get Started

<a href="https://colab.research.google.com/github/jina-ai/discoart/blob/main/discoart.ipynb"><img src="https://img.shields.io/badge/Open-in%20Colab-brightgreen?logo=google-colab&style=flat-square" alt="Open in Google Colab"/></a>

Note, GPU is required.

### Create artworks

```python
from discoart import create

da = create()
```

That's it! It will create with the [default text prompts and parameters](./discoart/resources/default.yml).

![](.github/create-demo.gif)

### Set prompts and parameters

Supported parameters are [listed here](./discoart/resources/default.yml). You can specify them in `create()`:

```python
from discoart import create

da = create(
    text_prompts='A painting of sea cliffs in a tumultuous storm, Trending on ArtStation.',
    init_image='https://d2vyhzeko0lke5.cloudfront.net/2f4f6dfa5a05e078469ebe57e77b72f0.png',
    skip_steps=100,
)
```

![](.github/parameter-demo.gif)

In case you forgot a parameter, just lookup the cheatsheet at anytime:

```python
from discoart import cheatsheet

cheatsheet()
```

The difference on the parameters between DiscoArt and DD5.x [is explained here](#whats-next). 


### Visualize results

`create()` returns `da`, a [DocumentArray](https://docarray.jina.ai/fundamentals/documentarray/)-type object. It contains the following information:
- All arguments passed to `create()` function, including seed, text prompts and model parameters.
- 4 generated image and its intermediate steps' images, where `4` is determined by `n_batches` and is the default value. 

This allows you to further post-process, analyze, export the results with powerful DocArray API.

Images are stored as Data URI in `.uri`, to save the first image as a local file:

```python
da[0].save_uri_to_file('discoart-result.png')
```

To save all final images:

```python
for idx, d in enumerate(da):
    d.save_uri_to_file(f'discoart-result-{idx}.png')
```

You can also display all four final images in a grid:

```python
da.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
```
![](.github/all-results.png)

Or display them one by one:

```python
for d in da:
    d.display()
```

Or take one particular run:

```python
da[0].display()
```

![](.github/display.png)

### Visualize intermediate steps

You can also zoom into a run (say the first run) and check out intermediate steps:

```python
da[0].chunks.plot_image_sprites(
    skip_empty=True, show_index=True, keep_aspect_ratio=True
)
```
![](.github/chunks.png)

You can `.display()` the chunks one by one, or save one via `.save_uri_to_file()`, or save all intermediate steps as a GIF:

```python
da[0].chunks.save_gif(
    'lighthouse.gif', show_index=True, inline_display=True, size_ratio=0.5
)
```

![](.github/lighthouse.gif)

### Export configs

You can review its parameters from `da[0].tags` or export it as an SVG image:

```python
from discoart.config import save_config_svg

save_config_svg(da)
```

![](.github/discoart-3205998582.svg)

### Pull results anywhere anytime

If you are a free-tier Google Colab user, one annoy thing is the lost of sessions from time to time. Or sometimes you just early stop the run as the first image is not good enough, and a keyboard interrupt will prevent `.create()` to return any result. Either case, you can easily recover the results by pulling the last session ID.

1. Find the session ID. It appears on top of the image. 
![](.github/session-id.png)

2. Pull the result via that ID **on any machine at any time**, not necessarily on Google Colab:
    ```python
    from docarray import DocumentArray

    da = DocumentArray.pull('discoart-3205998582')
    ```

### Reuse a Document as initial state

Consider a Document as a self-contained data with config and image, one can use it as the initial state for the future run. Its `.tags` will be used as the initial parameters; `.uri` if presented will be used as the initial image.

```python
from discoart import create
from docarray import DocumentArray

da = DocumentArray.pull('discoart-3205998582')

create(
    init_document=da[0],
    cut_ic_pow=0.5,
    tv_scale=600,
    cut_overview='[12]*1000',
    cut_innercut='[12]*1000',
    use_secondary_model=False,
)
```


### Environment variables

You can set environment variables to control the behavior of DiscoArt. The environment variables must be set before importing DiscoArt:

```bash
DISCOART_LOG_LEVEL='DEBUG' # more verbose logs
DISCOART_OPTOUT_CLOUD_BACKUP='1' # opt-out from cloud backup
DISCOART_DISABLE_IPYTHON='1' # disable ipython dependency
DISCOART_DISABLE_RESULT_SUMMARY='1' # disable result summary after the run ends
```

### Run in Docker

[![Docker Image Size (tag)](https://img.shields.io/docker/image-size/jinaai/discoart/latest?logo=docker&logoColor=white&style=flat-square)](https://hub.docker.com/repository/docker/jinaai/discoart)

We provide a prebuilt Docker image for running DiscoArt in the Jupyter Notebook. 

```bash
# docker build . -t jinaai/discoart  # if you want to build yourself
docker run -p 51000:8888 -v $(pwd):/home/jovyan/ -v $HOME/.cache:/root/.cache --gpus all jinaai/discoart
```



## What's next?

[Next is create](https://colab.research.google.com/github/jina-ai/discoart/blob/main/discoart.ipynb).

😎 **If you are already a DD user**: you are ready to go! There is no extra learning, DiscoArt respects the same parameter semantics as DD5.4. So just unleash your creativity!

There are some minor differences between DiscoArt and DD5.x:
  - DiscoArt fixes multiple bugs e.g. weighted prompts, cut scheduling in original DD5.4, which improves the generation quality.
  - DiscoArt does not support video generation and `image_prompt` (which was marked as ineffective in DD 5.4).
  - Due to no video support, `text_prompts` in DiscoArt accepts a string or a list of strings, not a dictionary; i.e. no frame index `0:` or `100:`.
  - `clip_models` accepts a list of values from [all open-clip pretrained models and weights](https://github.com/jina-ai/discoart/blob/main/discoart/resources/docstrings.yml#L90).
  - `cut_ic_pow` is changed to a scheduling parameter to control the power of inner cut, the syntax is the same as `cut_overview`, `cut_innercut`.
  - `clip_models_schedules` is added to control the scheduling of clip models, the syntax is the same as `cut_overview` but as a bool list `[True]*400+[False]*600`.
  - `diffusion_model` and `diffusion_model_config` can be specified load custom diffusion model and override the default diffusion model. 

👶 **If you are a [DALL·E Flow](https://github.com/jina-ai/dalle-flow/) or new user**: you may want to take step by step, as Disco Diffusion works in a very different way than DALL·E. It is much more advanced and powerful: e.g. Disco Diffusion can take weighted & structured text prompts; it can initialize from a image with controlled noise; and there are way more parameters one can tweak. Impatient prompt like `"armchair avocado"` will give you nothing but confusion and frustration. I highly recommend you to check out the following resources before trying your own prompt:
- [Zippy's Disco Diffusion Cheatsheet v0.3](https://docs.google.com/document/d/1l8s7uS2dGqjztYSjPpzlmXLjl5PM3IGkRWI3IiCuK7g/mobilebasic)
- [EZ Charts - Diffusion Parameter Studies](https://docs.google.com/document/d/1ORymHm0Te18qKiHnhcdgGp-WSt8ZkLZvow3raiu2DVU/edit#)
- [Disco Diffusion 70+ Artist Studies](https://weirdwonderfulai.art/resources/disco-diffusion-70-plus-artist-studies/)
- [A Traveler’s Guide to the Latent Space](https://sweet-hall-e72.notion.site/A-Traveler-s-Guide-to-the-Latent-Space-85efba7e5e6a40e5bd3cae980f30235f#e122e748b86e4fc0ad6a7a50e46d6e10)
- [Disco Diffusion Illustrated Settings](https://coar.notion.site/Disco-Diffusion-Illustrated-Settings-cd4badf06e08440c99d8a93d4cd39f51)
- [Coar’s Disco Diffusion Guide](https://coar.notion.site/coar/Coar-s-Disco-Diffusion-Guide-3d86d652c15d4ca986325e808bde06aa#8a3c6e9e4b6847afa56106eacb6f1f79)

<!-- start support-pitch -->
## Support

- Join our [Slack community](https://slack.jina.ai) and chat with other community members about ideas.
- Join our [Engineering All Hands](https://youtube.com/playlist?list=PL3UBBWOUVhFYRUa_gpYYKBqEAkO4sxmne) meet-up to discuss your use case and learn Jina's new features.
    - **When?** The second Tuesday of every month
    - **Where?**
      Zoom ([see our public events calendar](https://calendar.google.com/calendar/embed?src=c_1t5ogfp2d45v8fit981j08mcm4%40group.calendar.google.com&ctz=Europe%2FBerlin)/[.ical](https://calendar.google.com/calendar/ical/c_1t5ogfp2d45v8fit981j08mcm4%40group.calendar.google.com/public/basic.ics))
      and [live stream on YouTube](https://youtube.com/c/jina-ai)
- Subscribe to the latest video tutorials on our [YouTube channel](https://youtube.com/c/jina-ai)

## Join Us

DiscoArt is backed by [Jina AI](https://jina.ai) and licensed under [MIT License](./LICENSE). [We are actively hiring](https://jobs.jina.ai) AI engineers, solution engineers to build the next neural search ecosystem in open-source.

<!-- end support-pitch -->
