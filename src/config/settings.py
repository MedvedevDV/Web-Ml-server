import locale
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    models_dir: str
    cpu_cores: int
    max_loaded_models: int
    database_url: str

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / '.env',
        env_file_encoding='utf-8'
    )
    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent

settings = Settings()