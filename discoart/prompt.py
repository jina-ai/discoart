from types import SimpleNamespace
from typing import List, Dict, Any, Union

from .helper import PromptParser, _eval_scheduling_str


class PromptPlanner:
    def __init__(self, args):
        pmp = PromptParser(on_misspelled_token=args.on_misspelled_token)
        prompts = []  # type: List[Union[Dict[str, Any], SimpleNamespace]]
        if isinstance(args.text_prompts, list):
            # for legacy prompts
            for _p in args.text_prompts:
                _p = pmp.parse(_p)
                prompts.append({'tokenized': _p[0], 'weight': _p[1]})
        elif isinstance(args.text_prompts, dict):
            if args.text_prompts.get('version') == '1':
                prompts = args.text_prompts['prompts']
                for _p in args.text_prompts['prompts']:
                    txt, weight = pmp.parse(_p['text'])
                    weight2 = _p.get('weight', 1)
                    if weight != weight2:
                        raise ValueError(
                            f'weight is defined twice for the prompt `{_p["text"]}`: {weight} != {weight2}'
                        )
                    _p['tokenized'] = txt
                    _p['weight'] = weight
            else:
                raise ValueError(
                    f'unsupported text prompts schema: {args.text_prompts.get("version")}'
                )
        else:
            raise TypeError(f'unsupported text prompts type: {type(args.text_prompts)}')

        if not prompts:
            raise ValueError('no prompts found')

        # unify and set default for all prompts
        for idx, p in enumerate(prompts):
            p['steps'] = _eval_scheduling_str(p.get('steps', True))
            p['is_fuzzy'] = p.get('is_fuzzy', False)

            p['clip_guidance'] = set(p.get('clip_guidance', args.clip_models))
            if not set(p['clip_guidance']).issubset(args.clip_models):
                raise ValueError(
                    f'`clip_guidance` contains unknown clip models: {p["clip_guidance"]}'
                )
            prompts[idx] = SimpleNamespace(**p)

        # sum_weight = sum(p.weight for p in prompts)
        # if sum_weight <= 0:
        #     raise ValueError(
        #         f'The sum of all weights in the prompts must be strictly positive but get {sum_weight}'
        #     )
        self.prompts = prompts

    def get_prompt_ids(self, active_clip, num_step) -> List[int]:
        return [
            idx
            for idx, p in enumerate(self)
            if p.steps[num_step] and active_clip in p.clip_guidance
        ]

    def __iter__(self):
        return iter(self.prompts)
