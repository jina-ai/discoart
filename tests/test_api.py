from discoart import create, cheatsheet


def test_create():
    da = create(
        steps=1,
        n_batches=2,
        width_height=[64, 64],
        diffusion_model_config={'diffusion_steps': 100, 'timestep_respacing': 'ddim5'},
    )
    assert len(da) == 2
    assert da[0].uri


def test_cheatsheet():
    cheatsheet()
