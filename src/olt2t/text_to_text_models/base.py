from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Dict, List

from ..models import ChatRoleType, ChatTurn, StrT, SummaryTargetLength


class BaseTextToTextModel(ABC):
    @abstractmethod
    def respond(self, turns: List[ChatTurn]) -> Generator[StrT, None, None]:
        pass

    @abstractmethod
    def summarize(self, text: StrT, target_length: SummaryTargetLength) -> Generator[StrT, None, None]:
        pass


class BasicTextToTextModel(BaseTextToTextModel):
    def respond(self, turns: List[ChatTurn]) -> Generator[StrT, None, None]:
        if all(turn.role != ChatRoleType.SYSTEM for turn in turns):
            return self.respond([ChatTurn(role=ChatRoleType.SYSTEM, content="You are a helpful AI assistant.")] + turns)
        else:
            return self.generate([{"role": turn.role.value, "content": str(turn.content)} for turn in turns])

    def summarize(self, text: StrT, target_length: SummaryTargetLength) -> Generator[StrT, None, None]:
        return self.generate(
            [
                {
                    "role": "system",
                    "content": "Summarize the following text so that the resulting text will be length"
                    f" of {target_length}:",
                },
                {"role": "user", "content": str(text)},
            ]
        )

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> Generator[StrT, None, None]:
        pass
