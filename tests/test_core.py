from pytest_mock import MockerFixture

from olt2t.core import TextToTextCore
from olt2t.models import StrT
from olt2t.settings import TextToTextCoreSettings
from olt2t.text_to_text_models.base import BaseTextToTextModel


def test_core_create(mocker: MockerFixture) -> None:
    create_text_to_text_model = mocker.patch("olt2t.core.create_text_to_text_model")
    settings = TextToTextCoreSettings(text_to_text_model_settings={"type": "PHI_3", "path_or_model_name": "test"})
    actual = TextToTextCore.create(settings=settings)
    assert actual._model == create_text_to_text_model.return_value
    create_text_to_text_model.assert_called_once_with(settings=settings.text_to_text_model_settings)


def test_core_generate(mocker: MockerFixture) -> None:
    model = mocker.Mock(spec=BaseTextToTextModel)
    model.generate.return_value = iter([StrT("word1"), StrT("word2")])
    core = TextToTextCore(model=model)
    actual = list(core.generate(StrT("How to make a cake?")))
    assert actual == [StrT("word1"), StrT("word2")]
    model.generate.assert_called_once_with(StrT("How to make a cake?"))
