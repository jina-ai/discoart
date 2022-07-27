from types import SimpleNamespace

from discoart.config import default_args
from discoart.prompt import PromptPlanner


def test_prompt_builder_default_args():
    PromptPlanner(SimpleNamespace(**default_args))


def test_prompt_get_active_based_on_steps():
    smp = SimpleNamespace(**default_args)
    smp.text_prompts = {
        'version': '1',
        'prompts': [
            {'text': 'hello', 'schedule': '[True]*500+[False]*250+[True]*250'},
            {'text': 'bye', 'schedule': '[False]*500+[True]*500'},
        ],
    }
    pp = PromptPlanner(smp)
    for i in range(0, 500):
        assert pp.get_prompt_ids(smp.clip_models[0], i) == [0]
    for i in range(500, 750):
        assert pp.get_prompt_ids(smp.clip_models[0], i) == [1]
    for i in range(750, 1000):
        assert pp.get_prompt_ids(smp.clip_models[0], i) == [0, 1]


def test_prompt_get_active_based_on_clips():
    smp = SimpleNamespace(**default_args)
    smp.clip_models = set(['a', 'b', 'c'])
    smp.text_prompts = {
        'version': '1',
        'prompts': [
            {'text': 'hello', 'clip_guidance': ['a', 'b', 'c']},
            {'text': 'bye', 'clip_guidance': ['a']},
            {'text': 'bye', 'clip_guidance': ['c']},
        ],
    }
    pp = PromptPlanner(smp)
    for i in range(0, 1000):
        assert pp.get_prompt_ids('a', i) == [0, 1]
        assert pp.get_prompt_ids('c', i) == [0, 2]
        assert pp.get_prompt_ids('b', i) == [0]


def test_prompt_get_active_based_on_clips_steps():
    smp = SimpleNamespace(**default_args)
    smp.clip_models = set(['a', 'b', 'c'])
    smp.text_prompts = {
        'version': '1',
        'prompts': [
            {
                'text': 'hello',
                'clip_guidance': ['a', 'b', 'c'],
                'schedule': '[True]*500+[False]*500',
            },
            {'text': 'bye', 'clip_guidance': ['a'], 'schedule': '[True]*1000'},
            {
                'text': 'world',
                'clip_guidance': ['c'],
                'schedule': '[True]*400+[False]*300+[True]*300',
            },
        ],
    }
    pp = PromptPlanner(smp)
    for i in range(0, 1000):
        if i < 500:
            assert pp.get_prompt_ids('a', i) == [0, 1]
            assert pp.get_prompt_ids('b', i) == [0]
        if i >= 500:
            assert pp.get_prompt_ids('a', i) == [1]
            assert pp.get_prompt_ids('b', i) == []
        if i < 400:
            assert pp.get_prompt_ids('c', i) == [0, 2]
        if 400 <= i < 500:
            assert pp.get_prompt_ids('c', i) == [0]
        if 700 <= i < 1000:
            assert pp.get_prompt_ids('c', i) == [2]
