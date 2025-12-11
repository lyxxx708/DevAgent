from config.settings import Settings, settings


def test_settings_defaults_and_env_overrides():
    custom = Settings(llm_api_key="test-key")

    assert custom.data_dir == "./.devagent_data"
    assert custom.vector_dim == 768
    assert custom.coarse_k == 50
    assert custom.rerank_k == 20
    assert isinstance(settings, Settings)
