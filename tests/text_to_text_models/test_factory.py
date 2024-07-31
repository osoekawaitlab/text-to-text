import pytest
from pytest_mock import MockerFixture

from olt2t.settings import (
    BaseTextToTextModelSettings,
    EchoTextToTextModelSettings,
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


def test_create_text_to_text_model_generates_echo_model(mocker: MockerFixture) -> None:
    EchoTextToTextModel = mocker.patch("olt2t.text_to_text_models.factory.EchoTextToTextModel")
    settings = EchoTextToTextModelSettings()
    actual = create_text_to_text_model(settings=settings)
    assert actual == EchoTextToTextModel.return_value
    EchoTextToTextModel.assert_called_once_with()


def test_create_text_to_text_model_raises_value_error() -> None:
    settings = BaseTextToTextModelSettings(type=TextToTextModelType.ECHO)
    with pytest.raises(ValueError):
        create_text_to_text_model(settings=settings)  # type: ignore[arg-type]
