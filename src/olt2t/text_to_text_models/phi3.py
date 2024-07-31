from collections.abc import Generator

from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

from olt2t.models import StrT

from ..settings import PathOrModelName
from .base import BaseTextToTextModel


class Phi3TextToTextModel(BaseTextToTextModel):
    def __init__(self, path_or_model_name: PathOrModelName) -> None:
        self._path_or_model_name = path_or_model_name
        self._pipeline: Pipeline | None = None

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            model = AutoModelForCausalLM.from_pretrained(self._path_or_model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(self._path_or_model_name)
            self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return self._pipeline

    def generate(self, text: StrT) -> Generator[StrT, None, None]:
        for generated in self.pipeline(
            [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": str(text)},
            ],
            max_new_tokens=500,
            return_full_text=False,
            do_sample=False,
        ):
            yield StrT(generated["generated_text"])
