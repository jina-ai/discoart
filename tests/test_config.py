from discoart.config import default_args, save_config, load_config


def test_export_load_config(tmpfile):
    default_config = load_config(default_args)
    save_config(default_config, tmpfile)
    rec = load_config(tmpfile)
    assert rec == default_config
