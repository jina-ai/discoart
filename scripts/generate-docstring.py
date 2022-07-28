import re

import yaml
from yaml import Loader

with open('../discoart/resources/default.yml') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=Loader)
with open('../discoart/resources/docstrings.yml') as ymlfile:
    docs = yaml.load(ymlfile, Loader=Loader)

all_args = ['@overload\ndef create(']
cfg['init_document'] = None
for k, v in sorted(cfg.items(), key=lambda v: v[0]):
    v_type = type(v).__name__
    if k == 'init_image':
        v_type = 'str'
    elif k == 'init_document':
        v_type = 'Union[\'Document\', \'DocumentArray\']'
    elif k in ('seed', 'display_rate'):
        v_type = 'int'
    elif k in ('text_prompts', 'clip_models'):
        v_type = 'List[str]'
    elif k in (
        'cut_overview',
        'cut_innercut',
        'cut_icgray_p',
        'cut_ic_pow',
        'use_secondary_model',
        'cutn_batches',
        'skip_augs',
        'clip_guidance_scale',
        'tv_scale',
        'range_scale',
        'sat_scale',
        'init_scale',
        'clamp_grad',
        'clamp_max',
    ):
        if v_type == 'str':
            v_type = 'float'
        v_type = f'Union[{v_type}, str]'
    elif k == 'cut_schedules_group':
        v_type = 'str'
    elif k in ('batch_name', 'name_docarray'):
        v_type = 'str'
    elif k in ('skip_event', 'stop_event'):
        v_type = (
            'Union[\'multiprocessing.Event\', \'asyncio.Event\', \'threading.Event\']'
        )
    elif k == 'width_height':
        v_type = 'List[int]'
    elif k == 'transformation_percent':
        v_type = 'List[float]'
    elif k == 'clip_models_schedules':
        v_type = 'Dict[str, Union[str, List[str]]]'
    elif k == 'diffusion_model_config':
        v_type = 'Dict[str, Any]'
    if isinstance(v, str):
        v = f'\'{v}\''
    all_args.append(f'{k}: Optional[{v_type}] = {v},')
all_args.append(') -> Optional[\'DocumentArray\']:')
all_args.append('\n    ...')
func_signature = '\n'.join(all_args)

all_args = ["def create(**kwargs) -> Optional['DocumentArray']:"]
all_args.append('"""')
all_args.append(
    'Create Disco Diffusion artworks and return the result as a DocumentArray object.\n'
)
for k, v in sorted(docs.items(), key=lambda x: x[0]):
    if v:
        _v = v.replace("\n", "").strip()
        all_args.append(f':param {k}: {_v}')
all_args.append(':return: a DocumentArray object that has `n_batches` Documents')
all_args.append('"""')
indent = '    '
func_docstring = f'\n{indent}'.join(all_args)

src_py = '../discoart/create.py'
with open(src_py) as fp:
    _old = fp.read()
    _old = re.sub(
        r'(# begin_create_overload\s*?\n).*(\n\s*?# end_create_overload)',
        rf'\g<1>{func_signature}\g<2>',
        _old,
        flags=re.DOTALL,
    )
    old = re.sub(
        r'(# begin_create_docstring\s*?\n).*(\n\s*?# end_create_docstring)',
        rf'\g<1>{func_docstring}\g<2>',
        _old,
        flags=re.DOTALL,
    )

with open(src_py, 'w') as fp:
    fp.write(old)

# final_str = f'@overload\n{signature_str}\n{indent}"""{doc_str_title}\n\n{doc_str}{return_str}\n\n{noqa_str}\n{indent}"""'
# final_code = re.sub(
#     rf'(# overload_inject_start_{regex_tag or cli_entrypoint}).*(# overload_inject_end_{regex_tag or cli_entrypoint})',
#     f'\\1\n{final_str}\n{indent}\\2',
#     open(filepath).read(),
#     0,
#     re.DOTALL,
# )
