import os
from collections.abc import Generator

import pytest
from pytest_mock import MockerFixture

from olt2t import StrT
from olt2t.models import ChatTurn
from olt2t.settings import ApiKey, OpenAiModelType
from olt2t.text_to_text_models.openai import OpenAiTextToTextModel


@pytest.mark.openai_api_key_required
def test_openai_text_to_text_model_generate() -> None:
    sut = OpenAiTextToTextModel(
        api_key=ApiKey.from_str(os.environ.get("OPENAI_API_KEY", "")), openai_model_type=OpenAiModelType.GPT_4O_MINI
    )
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How to make a cake?"},
        {"role": "assistant", "content": "I don't know."},
        {"role": "user", "content": "Okay."},
    ]
    result = sut.generate(messages)
    assert isinstance(result, Generator)
    actual = list(result)
    assert len(actual) > 0
    assert all(isinstance(item, StrT) for item in actual)


def test_openai_text_to_text_model_summarize(mocker: MockerFixture) -> None:
    sut = OpenAiTextToTextModel(api_key=ApiKey.from_str("DUMMY_KEY"), openai_model_type=OpenAiModelType.GPT_4O_MINI)
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


def test_openai_text_to_text_respond(mocker: MockerFixture) -> None:
    sut = OpenAiTextToTextModel(api_key=ApiKey.from_str("DUMMY_KEY"), openai_model_type=OpenAiModelType.GPT_4O_MINI)
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
