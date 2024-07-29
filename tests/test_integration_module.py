from collections.abc import Generator

import pytest

import olt2t


@pytest.mark.slow
def test_olt2t_as_module_generate() -> None:
    settings = olt2t.TextToTextCoreSettings(
        text_to_text_model_settings={"type": "PHI_3", "path_or_model_name": "microsoft/Phi-3-mini-4k-instruct"}
    )
    core = olt2t.TextToTextCore.create(settings=settings)
    actual = core.generate(olt2t.StrT("How to make a cake?"))
    assert isinstance(actual, Generator)
    words = list(actual)
    assert len(words) > 0
