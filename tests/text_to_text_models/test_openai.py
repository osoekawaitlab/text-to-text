import os
from collections.abc import Generator

import pytest

from olt2t import StrT
from olt2t.settings import ApiKey, OpenAiModelType
from olt2t.text_to_text_models.openai import OpenAiTextToTextModel


@pytest.mark.openai_api_key_required
def test_openai_text_to_text_model_generate() -> None:
    sut = OpenAiTextToTextModel(
        api_key=ApiKey.from_str(os.environ.get("OPENAI_API_KEY", "")), openai_model_type=OpenAiModelType.GPT_4O_MINI
    )
    text = StrT.from_str("What is the meaning of life?")
    result = sut.generate(text)
    assert isinstance(result, Generator)
    actual = list(result)
    assert len(actual) > 0
    assert all(isinstance(item, StrT) for item in actual)
