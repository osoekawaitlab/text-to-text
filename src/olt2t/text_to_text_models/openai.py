from collections.abc import Generator

from openai import OpenAI

from ..models import StrT
from ..settings import ApiKey, OpenAiModelType
from .base import BaseTextToTextModel


class OpenAiTextToTextModel(BaseTextToTextModel):
    def __init__(self, api_key: ApiKey, openai_model_type: OpenAiModelType) -> None:
        self._api_key = api_key
        self._openai_model_type = openai_model_type
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=str(self._api_key))
        return self._client

    def generate(self, text: StrT) -> Generator[StrT, None, None]:
        for generated in self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": str(text)},
            ],
            model=self._openai_model_type.value,
            stream=True,
        ):
            res = generated.choices[0].delta.content
            if res is not None:
                yield StrT(res)
