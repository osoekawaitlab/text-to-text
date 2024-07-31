from collections.abc import Generator

import olt2t


def test_olt2t_as_module_generate() -> None:
    settings = olt2t.TextToTextCoreSettings(text_to_text_model_settings={"type": "ECHO"})
    core = olt2t.TextToTextCore.create(settings=settings)
    actual = core.generate(olt2t.StrT("How to make a cake?"))
    assert isinstance(actual, Generator)
    words = list(actual)
    assert words == [olt2t.StrT("How to make a cake?")]
