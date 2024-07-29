import re

import olt2t


def test_olt2t_has_version() -> None:
    assert re.match(r"\d+\.\d+\.\d+", olt2t.__version__)


def test_olt2t_has_text_to_text_core_settings() -> None:
    assert olt2t.TextToTextCoreSettings == olt2t.settings.TextToTextCoreSettings


def test_olt2t_has_text_to_text_core() -> None:
    assert olt2t.TextToTextCore == olt2t.core.TextToTextCore
