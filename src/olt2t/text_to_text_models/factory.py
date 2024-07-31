from ..settings import (
    EchoTextToTextModelSettings,
    OpenAiTextToTextModelSettings,
    Phi3TextToTextModelSettings,
    TextToTextModelSettings,
)
from .base import BaseTextToTextModel
from .echo import EchoTextToTextModel
from .openai import OpenAiTextToTextModel
from .phi3 import Phi3TextToTextModel


def create_text_to_text_model(settings: TextToTextModelSettings) -> BaseTextToTextModel:
    if isinstance(settings, Phi3TextToTextModelSettings):
        return Phi3TextToTextModel(path_or_model_name=settings.path_or_model_name)
    if isinstance(settings, EchoTextToTextModelSettings):
        return EchoTextToTextModel()
    if isinstance(settings, OpenAiTextToTextModelSettings):
        return OpenAiTextToTextModel(api_key=settings.api_key, openai_model_type=settings.openai_model_type)
    raise ValueError(f"Unsupported model type: {settings.type}")
