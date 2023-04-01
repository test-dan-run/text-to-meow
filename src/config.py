from pydantic import BaseSettings, Field

class BaseConfig(BaseSettings):
    """Define any config here.
    See here for documentation:
    https://pydantic-docs.helpmanual.io/usage/settings/
    """
    # KNative assigns a $PORT environment variable to the container
    port: int = Field(default=8080, env="PORT",description="Gradio App Server Port")

    manifest_path: str = 'meows/manifest.json'
    sample_rate: int = 16000
    init_factor: float = 0.3
    add_factor: float = 0.2
    power_factor: float = 0.8

config = BaseConfig()