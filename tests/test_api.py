import pytest

from discoart import create, cheatsheet


@pytest.fixture
def mini_config():
    yield dict(
        steps=1,
        n_batches=2,
        width_height=[64, 64],
        diffusion_model_config={'diffusion_steps': 25, 'timestep_respacing': 'ddim5'},
        batch_name='cicd',
    )


def test_create(mini_config):
    da = create(**mini_config)
    assert len(da) == 2
    assert da[0].uri


def test_cheatsheet():
    cheatsheet()
