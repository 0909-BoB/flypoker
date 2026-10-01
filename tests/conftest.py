import pytest


@pytest.fixture(autouse=True)
def isolated_learning_dir(tmp_path, monkeypatch):
    """Games run by tests must not write into the real runs/ folder."""
    from flypoker import learning

    monkeypatch.setenv("FLYPOKER_LEARN_DIR", str(tmp_path))
    monkeypatch.setattr(learning, "_hub", None)
    yield
