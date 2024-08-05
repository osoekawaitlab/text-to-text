import pytest
from pytest_mock import MockerFixture

from olt2t.settings import (
    BaseTextToTextModelSettings,
    OpenAiModelType,
    OpenAiTextToTextModelSettings,
    PathOrModelName,
    Phi3TextToTextModelSettings,
    TextToTextModelType,
)
from olt2t.text_to_text_models.factory import create_text_to_text_model


def test_create_text_to_text_model_generates_phi3_model(mocker: MockerFixture) -> None:
    Phi3TextToTextModel = mocker.patch("olt2t.text_to_text_models.factory.Phi3TextToTextModel")
    settings = Phi3TextToTextModelSettings(path_or_model_name="path")
    actual = create_text_to_text_model(settings=settings)
    assert actual == Phi3TextToTextModel.return_value
    Phi3TextToTextModel.assert_called_once_with(path_or_model_name=PathOrModelName("path"))


def test_create_text_to_text_model_generates_openai_model(mocker: MockerFixture) -> None:
    OpenAiTextToTextModel = mocker.patch("olt2t.text_to_text_models.factory.OpenAiTextToTextModel")
    settings = OpenAiTextToTextModelSettings(api_key="key", openai_model_type=OpenAiModelType.GPT_4O_MINI)
    actual = create_text_to_text_model(settings=settings)
    assert actual == OpenAiTextToTextModel.return_value
    OpenAiTextToTextModel.assert_called_once_with(api_key="key", openai_model_type=OpenAiModelType.GPT_4O_MINI)


def test_create_text_to_text_model_raises_value_error() -> None:
    settings = BaseTextToTextModelSettings(type=TextToTextModelType.PHI_3)
    with pytest.raises(ValueError):
        create_text_to_text_model(settings=settings)  # type: ignore[arg-type]
