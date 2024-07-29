from collections.abc import Generator

from .models import StrT
from .settings import TextToTextCoreSettings
from .text_to_text_models.base import BaseTextToTextModel
from .text_to_text_models.factory import create_text_to_text_model


class TextToTextCore:
    def __init__(self, model: BaseTextToTextModel) -> None:
        self._model = model

    @classmethod
    def create(cls, settings: TextToTextCoreSettings) -> "TextToTextCore":
        model = create_text_to_text_model(settings=settings.text_to_text_model_settings)
        return cls(model=model)

    def generate(self, text: StrT) -> Generator[StrT, None, None]:
        return self._model.generate(text)
