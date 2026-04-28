import os
from omegaconf import OmegaConf


class ConfigReader:
    __instance = None
    __initialized = False

    def __init__(self):
        if not ConfigReader.__initialized:
            config_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', 'config', 'train_conf.yaml')
            )
            if not os.path.exists(config_path):
                raise FileNotFoundError(f'Config not found: {config_path}')
            self.config = OmegaConf.load(config_path)
            ConfigReader.__initialized = True

    def __new__(cls):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def get(self, theme: str, key: str, default=None):
        return OmegaConf.select(self.config, f'{theme}.{key}', default=default)

    def get_section(self, theme: str):
        return self.config.get(theme)

    @property
    def raw_config(self):
        return self.config
