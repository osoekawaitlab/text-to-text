from collections.abc import Generator

import pytest
from pytest_mock import MockerFixture

from olt2t import StrT
from olt2t.models import ChatTurn
from olt2t.settings import PathOrModelName
from olt2t.text_to_text_models.phi3 import Phi3TextToTextModel


@pytest.mark.slow
def test_phi3_text_to_text_model_generate() -> None:
    sut = Phi3TextToTextModel(path_or_model_name=PathOrModelName("microsoft/Phi-3-mini-4k-instruct"))
    actual = sut.generate(
        [
            {"role": "system", "content": "You are a bot which can answer only 'yes' or 'no'."},
            {"role": "user", "content": "Can you answer only 'yes' or 'no'?"},
        ]
    )
    assert isinstance(actual, Generator)
    words = list(actual)
    assert len(words) >= 1


def test_phi3_text_to_text_model_summarize(mocker: MockerFixture) -> None:
    sut = Phi3TextToTextModel(path_or_model_name=PathOrModelName("microsoft/Phi-3-mini-4k-instruct"))
    mocker.patch.object(sut, "generate", return_value=iter([StrT("word1"), StrT("word2")]))
    actual = sut.summarize("How to make a cake?", target_length=5)
    actual_list = list(actual)
    assert actual_list == [StrT("word1"), StrT("word2")]
    sut.generate.assert_called_once_with(
        [
            {
                "role": "system",
                "content": "Summarize the following text so that the resulting text will be length of 5:",
            },
            {"role": "user", "content": "How to make a cake?"},
        ]
    )


def test_phi3_text_to_text_model_respond(mocker: MockerFixture) -> None:
    sut = Phi3TextToTextModel(path_or_model_name=PathOrModelName("microsoft/Phi-3-mini-4k-instruct"))
    mocker.patch.object(sut, "generate", return_value=iter([StrT("word1"), StrT("word2")]))
    actual = sut.respond(
        [
            ChatTurn(role="user", content="How to make a cake?"),
            ChatTurn(role="assistant", content="I don't know."),
            ChatTurn(role="user", content="Okay."),
        ]
    )
    actual_list = list(actual)
    assert actual_list == [StrT("word1"), StrT("word2")]
    sut.generate.assert_called_once_with(
        [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "How to make a cake?"},
            {"role": "assistant", "content": "I don't know."},
            {"role": "user", "content": "Okay."},
        ]
    )
