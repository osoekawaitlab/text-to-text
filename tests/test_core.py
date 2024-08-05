from pytest_mock import MockerFixture

from olt2t.core import TextToTextCore
from olt2t.models import ChatRoleType, ChatTask, ChatTurn, StrT, SummarizationTask
from olt2t.settings import TextToTextCoreSettings
from olt2t.text_to_text_models.base import BaseTextToTextModel


def test_core_create(mocker: MockerFixture) -> None:
    create_text_to_text_model = mocker.patch("olt2t.core.create_text_to_text_model")
    settings = TextToTextCoreSettings(text_to_text_model_settings={"type": "PHI_3", "path_or_model_name": "test"})
    actual = TextToTextCore.create(settings=settings)
    assert actual._model == create_text_to_text_model.return_value
    create_text_to_text_model.assert_called_once_with(settings=settings.text_to_text_model_settings)


def test_core_chat(mocker: MockerFixture) -> None:
    model = mocker.Mock(spec=BaseTextToTextModel)
    model.respond.return_value = iter([StrT("word1"), StrT("word2")])
    task = ChatTask(
        turns=[
            ChatTurn(role=ChatRoleType.USER, content=StrT("How to make a cake?")),
            ChatTurn(role=ChatRoleType.ASSISTANT, content=StrT("I don't know.")),
            ChatTurn(role=ChatRoleType.USER, content=StrT("Okay.")),
        ]
    )
    core = TextToTextCore(model=model)
    actual = list(core(task))
    assert actual == [StrT("word1"), StrT("word2")]
    model.respond.assert_called_once_with(task.turns)


def test_core_summarize(mocker: MockerFixture) -> None:
    model = mocker.Mock(spec=BaseTextToTextModel)
    model.summarize.return_value = iter([StrT("word1")])
    task = SummarizationTask(text=StrT("How to make a cake?"), target_length=5)
    core = TextToTextCore(model=model)
    actual = list(core(task))
    assert actual == [StrT("word1")]
    model.summarize.assert_called_once_with(StrT("How to make a cake?"), target_length=5)
