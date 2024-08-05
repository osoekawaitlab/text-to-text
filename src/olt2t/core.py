from collections.abc import Generator

from .models import ChatTask, StrT, SummarizationTask, Task
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

    @property
    def model(self) -> BaseTextToTextModel:
        return self._model

    def __call__(self, task: Task) -> Generator[StrT, None, None]:
        if isinstance(task, SummarizationTask):
            return self.model.summarize(task.text, target_length=task.target_length)
        elif isinstance(task, ChatTask):
            return self.model.respond(task.turns)
        raise TypeError(f"Unsupported task type: {type(task)}")
