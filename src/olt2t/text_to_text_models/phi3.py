from collections.abc import Generator

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ..models import StrT
from ..settings import PathOrModelName
from .base import BaseTextToTextModel


class Phi3TextToTextModel(BaseTextToTextModel):
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

    def generate(self, text: StrT) -> Generator[StrT, None, None]:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": str(text)},
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        input_length = inputs.input_ids.shape[1]

        for outputs in self.model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        ):
            new_tokens = outputs[input_length:]
            new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            if new_text.strip():
                yield StrT(new_text)
