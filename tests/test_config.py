from discoart.config import default_args, save_config, load_config, export_python
from discoart.helper import _eval_scheduling_str


def test_export_load_config(tmpfile):
    default_config = load_config(default_args)
    save_config(default_config, tmpfile)
    rec = load_config(tmpfile)
    assert rec == default_config


def test_format_config():
    default_args['name_docarray'] = 'test-{steps}'
    default_config = load_config(default_args)
    assert default_config['name_docarray'] == f'test-{default_args["steps"]}'

    default_args['name_docarray'] = 'test-{steps}-{clip_models}'
    default_config = load_config(default_args)
    assert (
        default_config['name_docarray']
        == f'test-{default_args["steps"]}-{default_args["clip_models"]}'
    )


def test_export_python():
    assert export_python(default_args)


def test_eval_schedule_string():
    assert _eval_scheduling_str('1') == [1] * 1000
    assert _eval_scheduling_str('[1] * 1000') == [1] * 1000
    assert _eval_scheduling_str(1) == [1] * 1000
    assert _eval_scheduling_str('1.') == [1] * 1000
    assert _eval_scheduling_str('True') == [True] * 1000
    assert _eval_scheduling_str('False') == [False] * 1000
    assert _eval_scheduling_str(True) == [True] * 1000
