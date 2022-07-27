import copy
from types import SimpleNamespace
from typing import List, Dict, Any, Union

from .helper import PromptParser, _eval_scheduling_str


class PromptPlanner:
    def __init__(self, args):
        text_prompts = copy.deepcopy(args.text_prompts)
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        pmp = PromptParser(on_misspelled_token=args.on_misspelled_token)
        prompts = []  # type: List[Union[Dict[str, Any], SimpleNamespace]]
        if isinstance(text_prompts, list):
            # for legacy prompts
            for _p in text_prompts:
                _pw = pmp.parse(_p)
                prompts.append({'tokenized': _pw[0], 'weight': _pw[1]})
        elif isinstance(text_prompts, dict):
            if text_prompts.get('version') == '1':
                prompts = text_prompts['prompts']
                for _p in text_prompts['prompts']:
                    txt, weight = pmp.parse(_p['text'])
                    weight = _p.get('weight', weight)
                    _p['tokenized'] = txt
                    _p['weight'] = weight
            else:
                raise ValueError(
                    f'unsupported text prompts schema: {text_prompts.get("version")}'
                )
        else:
            raise TypeError(f'unsupported text prompts type: {type(text_prompts)}')

        if not prompts:
            raise ValueError('no prompts found')

        # unify and set default for all prompts
        for idx, p in enumerate(prompts):
            p['schedule'] = _eval_scheduling_str(p.get('schedule', True))
            p['clip_guidance'] = set(p.get('clip_guidance', args.clip_models))
            if not set(p['clip_guidance']).issubset(args.clip_models):
                raise ValueError(
                    f'`clip_guidance` contains unknown clip models: {p["clip_guidance"]}'
                )
            prompts[idx] = SimpleNamespace(**p)

        self.prompts = prompts

    def get_prompt_ids(self, active_clip, num_step) -> List[int]:
        return [
            idx
            for idx, p in enumerate(self)
            if p.schedule[num_step] and active_clip in p.clip_guidance
        ]

    def __iter__(self):
        return iter(self.prompts)
