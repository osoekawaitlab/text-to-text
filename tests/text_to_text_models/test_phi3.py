from collections.abc import Generator

import pytest

from olt2t import StrT
from olt2t.settings import PathOrModelName
from olt2t.text_to_text_models.phi3 import Phi3TextToTextModel


@pytest.mark.slow
def test_phi3_generate() -> None:
    sut = Phi3TextToTextModel(path_or_model_name=PathOrModelName("microsoft/Phi-3-mini-4k-instruct"))
    actual = sut.generate(StrT("Hello."))
    assert isinstance(actual, Generator)
    words = list(actual)
    assert len(words) > 1
