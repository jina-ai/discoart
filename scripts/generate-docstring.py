import yaml
from yaml import Loader

with open('../discoart/resources/default.yml') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=Loader)
with open('../discoart/resources/docstrings.yml') as ymlfile:
    docs = yaml.load(ymlfile, Loader=Loader)

all_args = ['@overload\ndef create(']
for k, v in cfg.items():
    v_type = type(v).__name__
    if k == 'init_image':
        v_type = 'str'
    elif k == 'seed':
        v_type = 'int'
    if v_type == 'str' and v is not None:
        v = f'\'{v}\''
    if k in ('text_prompts', 'clip_models'):
        v_type = 'List[str]'
    elif k == 'width_height':
        v_type = 'List[int]'
    elif k == 'transformation_percent':
        v_type = 'List[float]'
    elif k == 'clip_models_schedules':
        v_type = 'Dict[str, str]'
    elif k == 'diffusion_model_config':
        v_type = 'Dict[str, Any]'
    all_args.append(f'{k}: Optional[{v_type}] = {v},')
all_args.append(') -> Optional[\'DocumentArray\']:')
all_args.append('"""')
all_args.append(
    'Create Disco Diffusion artworks and return the result as a DocumentArray object.\n'
)
for k, v in docs.items():
    if v:
        _v = v.replace("\n", "").strip()
        all_args.append(f':param {k}: {_v}')
all_args.append(':return: a DocumentArray object that has `n_batches` Documents')
all_args.append('"""')
indent = '    '
print(f'\n{indent}'.join(all_args))


# final_str = f'@overload\n{signature_str}\n{indent}"""{doc_str_title}\n\n{doc_str}{return_str}\n\n{noqa_str}\n{indent}"""'
# final_code = re.sub(
#     rf'(# overload_inject_start_{regex_tag or cli_entrypoint}).*(# overload_inject_end_{regex_tag or cli_entrypoint})',
#     f'\\1\n{final_str}\n{indent}\\2',
#     open(filepath).read(),
#     0,
#     re.DOTALL,
# )
