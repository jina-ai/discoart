from discoart import create, cheatsheet


def test_create(mini_config):
    da = create(**mini_config)
    assert len(da) == 2
    assert da[0].uri


def test_cheatsheet():
    cheatsheet()


def test_export_svg(mini_config):
    from discoart.config import save_config_svg

    save_config_svg(mini_config)


def test_argparser_config():
    from discoart.parser import get_main_parser

    get_main_parser().parse_args(['config'])
