from enum import Enum
from typing import Annotated, Literal, TypeAlias, Union

from oltl import BaseModel as OltlBaseModel
from oltl import BaseString, LowerBoundIntegerMixIn
from pydantic import Field

StrT: TypeAlias = BaseString


class LanguageType(str, Enum):
    ENGLISH = "English"
    JAPANESE = "Japanese"


class TaskType(str, Enum):
    SUMMARIZATION = "SUMMARIZATION"
    CHAT = "CHAT"


class SummaryTargetLength(LowerBoundIntegerMixIn):
    @classmethod
    def get_min_value(self) -> int:
        return 1


class ChatRoleType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class BaseModel(OltlBaseModel):
    pass


class BaseTask(BaseModel):
    type: TaskType


class BaseSingleTextTextTask(BaseTask):
    text: StrT


class SummarizationTask(BaseSingleTextTextTask):
    type: Literal[TaskType.SUMMARIZATION] = TaskType.SUMMARIZATION
    target_length: SummaryTargetLength


class ChatTurn(BaseModel):
    role: ChatRoleType
    content: StrT


class ChatTask(BaseTask):
    type: Literal[TaskType.CHAT] = TaskType.CHAT
    turns: list[ChatTurn]


Task = Annotated[Union[SummarizationTask, ChatTask], Field(discriminator="type")]
