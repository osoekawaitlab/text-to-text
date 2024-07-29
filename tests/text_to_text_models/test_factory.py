from pytest_mock import MockerFixture

from olt2t.settings import PathOrModelName, Phi3TextToTextModelSettings
from olt2t.text_to_text_models.factory import create_text_to_text_model


def test_create_text_to_text_model_generates_phi3_model(mocker: MockerFixture) -> None:
    Phi3TextToTextModel = mocker.patch("olt2t.text_to_text_models.factory.Phi3TextToTextModel")
    settings = Phi3TextToTextModelSettings(path_or_model_name="path")
    actual = create_text_to_text_model(settings=settings)
    assert actual == Phi3TextToTextModel.return_value
    Phi3TextToTextModel.assert_called_once_with(path_or_model_name=PathOrModelName("path"))
