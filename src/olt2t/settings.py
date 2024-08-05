from enum import Enum
from typing import Annotated, Literal, Union

from oltl import NonEmptyStringMixIn, TrimmedStringMixIn
from oltl.settings import BaseSettings as OltlBaseSettings
from pydantic import Field
from pydantic_settings import SettingsConfigDict


class PathOrModelName(NonEmptyStringMixIn, TrimmedStringMixIn):
    pass


class ApiKey(NonEmptyStringMixIn, TrimmedStringMixIn):
    pass


class BaseSettings(OltlBaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLT2T_")


class TextToTextModelType(str, Enum):
    PHI_3 = "PHI_3"
    OPENAI = "OPENAI"


class OpenAiModelType(str, Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"


class BaseTextToTextModelSettings(BaseSettings):
    type: TextToTextModelType


class BaseHuggingFaceTextToTextModelSettings(BaseTextToTextModelSettings):
    path_or_model_name: PathOrModelName


class Phi3TextToTextModelSettings(BaseHuggingFaceTextToTextModelSettings):
    type: Literal[TextToTextModelType.PHI_3] = TextToTextModelType.PHI_3


class OpenAiTextToTextModelSettings(BaseTextToTextModelSettings):
    type: Literal[TextToTextModelType.OPENAI] = TextToTextModelType.OPENAI
    api_key: ApiKey
    openai_model_type: OpenAiModelType


TextToTextModelSettings = Annotated[
    Union[Phi3TextToTextModelSettings, OpenAiTextToTextModelSettings],
    Field(discriminator="type"),
]


class TextToTextCoreSettings(BaseSettings):
    text_to_text_model_settings: TextToTextModelSettings
