from discoart import create, cheatsheet


def test_create():
    da = create(steps=10, n_batches=2)
    assert len(da) == 2
    assert da[0].uri


def test_cheatsheet():
    cheatsheet()
