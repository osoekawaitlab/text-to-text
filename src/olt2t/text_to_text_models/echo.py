from collections.abc import Generator

from olt2t.models import StrT

from .base import BaseTextToTextModel


class EchoTextToTextModel(BaseTextToTextModel):
    def generate(self, text: StrT) -> Generator[StrT, None, None]:
        yield text
