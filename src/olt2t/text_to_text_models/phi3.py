from collections.abc import Generator

from olt2t.models import StrT

from ..settings import PathOrModelName
from .base import BaseTextToTextModel


class Phi3TextToTextModel(BaseTextToTextModel):
    def __init__(self, path_or_model_name: PathOrModelName) -> None:
        self._path_or_model_name = path_or_model_name

    def generate(self, text: StrT) -> Generator[StrT, None, None]:
        raise NotImplementedError
