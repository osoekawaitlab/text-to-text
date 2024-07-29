from ..settings import Phi3TextToTextModelSettings, TextToTextModelSettings
from .base import BaseTextToTextModel
from .phi3 import Phi3TextToTextModel


def create_text_to_text_model(settings: TextToTextModelSettings) -> BaseTextToTextModel:
    if isinstance(settings, Phi3TextToTextModelSettings):
        return Phi3TextToTextModel(path_or_model_name=settings.path_or_model_name)
