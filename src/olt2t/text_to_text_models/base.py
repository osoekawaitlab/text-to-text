from abc import ABC, abstractmethod
from collections.abc import Generator

from ..models import StrT


class BaseTextToTextModel(ABC):
    @abstractmethod
    def generate(self, text: StrT) -> Generator[StrT, None, None]:
        pass
