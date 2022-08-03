import pytest

from discoart.config import default_args, save_config, load_config, export_python
from discoart.helper import _eval_scheduling_str, _is_valid_schedule_str


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


@pytest.mark.parametrize(
    'val, expected',
    [
        ('[100]*600+[200]*400', True),
        ('[100]*600+[2.3]*400', True),
        ('[100]*600+[2.3]*400', True),
        ('1', True),
        ('True', True),
        ('Truetrue', False),
        ('False', True),
        ('true', False),
        ('sdd ds', False),
        ('[True, False]*1000', True),
        ('[True]*500+[False]*400', True),
        ('[0.5]*400+[0.2]*300+[True]*200', True),
        ('[hello]*1000', False),
        ('del a', False),
        ('([1]+[2])*50', True),
        ('[False,True,1,0.23,23,]*1000', True),
    ],
)
def test_chec_schedule_str(val, expected):
    assert _is_valid_schedule_str(val) == expected
