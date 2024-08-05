from collections.abc import Generator
from typing import Dict, List

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from ..models import StrT
from ..settings import ApiKey, OpenAiModelType
from .base import BasicTextToTextModel


class OpenAiTextToTextModel(BasicTextToTextModel):
    def __init__(self, api_key: ApiKey, openai_model_type: OpenAiModelType) -> None:
        self._api_key = api_key
        self._openai_model_type = openai_model_type
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=str(self._api_key))
        return self._client

    def generate(self, messages: List[Dict[str, str]]) -> Generator[StrT, None, None]:
        typed_messages: List[ChatCompletionMessageParam] = []
        for message in messages:
            if message["role"] == "system":
                typed_messages.append(ChatCompletionSystemMessageParam(role="system", content=message["content"]))
            elif message["role"] == "user":
                typed_messages.append(ChatCompletionUserMessageParam(role="user", content=message["content"]))
            elif message["role"] == "assistant":
                typed_messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=message["content"]))
            else:
                raise ValueError(f"Invalid role: {message['role']}")
        for generated in self.client.chat.completions.create(
            messages=typed_messages,
            model=self._openai_model_type.value,
            stream=True,
        ):
            res = generated.choices[0].delta.content
            if res is not None:
                yield StrT(res)
