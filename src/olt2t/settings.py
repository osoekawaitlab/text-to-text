from oltl.settings import BaseSettings as OltlBaseSettings
from pydantic_settings import SettingsConfigDict


class BaseSettings(OltlBaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLT2T_")
