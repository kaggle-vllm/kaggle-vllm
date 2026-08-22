from kaggle_vllm.environment import collect


def test_collect_returns_environment():
    env = collect()
    assert isinstance(env.python, str)
    assert isinstance(env.gpu_names, list)
    assert isinstance(env.gpu_caps, list)
