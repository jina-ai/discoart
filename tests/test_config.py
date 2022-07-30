from discoart.config import default_args, save_config, load_config


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
