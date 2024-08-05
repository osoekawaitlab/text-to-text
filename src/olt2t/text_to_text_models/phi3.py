from collections.abc import Generator
from threading import Thread
from typing import Dict, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)

from ..models import StrT
from ..settings import PathOrModelName
from .base import BasicTextToTextModel


class Phi3TextToTextModel(BasicTextToTextModel):
    def __init__(self, path_or_model_name: PathOrModelName) -> None:
        self._path_or_model_name = path_or_model_name
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(self._path_or_model_name, trust_remote_code=True)
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._path_or_model_name)
        return self._tokenizer

    def generate(self, messages: List[Dict[str, str]]) -> Generator[StrT, None, None]:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            inputs,
            max_new_tokens=500,
            do_sample=False,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if new_text.strip():
                yield StrT(new_text)

        thread.join()
