from enum import Enum
from typing import Annotated, Literal

from oltl import NonEmptyStringMixIn, TrimmedStringMixIn
from oltl.settings import BaseSettings as OltlBaseSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class PathOrModelName(NonEmptyStringMixIn, TrimmedStringMixIn):
    pass


class BaseSettings(OltlBaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLT2T_")


class TextToTextModelType(str, Enum):
    PHI_3 = "PHI_3"


class BaseTextToTextModelSettings(BaseSettings):
    type: TextToTextModelType


class BaseHuggingFaceTextToTextModelSettings(BaseTextToTextModelSettings):
    path_or_model_name: PathOrModelName


class Phi3TextToTextModelSettings(BaseHuggingFaceTextToTextModelSettings):
    type: Literal[TextToTextModelType.PHI_3] = TextToTextModelType.PHI_3


TextToTextModelSettings = Annotated[Phi3TextToTextModelSettings, Field(discriminator="type")]


class TextToTextCoreSettings(BaseSettings):
    text_to_text_model_settings: TextToTextModelSettings
