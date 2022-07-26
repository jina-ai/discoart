from types import SimpleNamespace

from discoart.config import default_args
from discoart.prompt import build_prompts


def test_prompt_builder_default_args():
    build_prompts(SimpleNamespace(**default_args))
