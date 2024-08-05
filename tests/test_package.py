import re

import olt2t


def test_olt2t_has_version() -> None:
    assert re.match(r"\d+\.\d+\.\d+", olt2t.__version__)


def test_olt2t_has_text_to_text_core_settings() -> None:
    assert olt2t.TextToTextCoreSettings == olt2t.settings.TextToTextCoreSettings


def test_olt2t_has_text_to_text_core() -> None:
    assert olt2t.TextToTextCore == olt2t.core.TextToTextCore


def test_olt2t_has_summarization_task() -> None:
    assert olt2t.SummarizationTask == olt2t.models.SummarizationTask


def test_olt2t_has_chat_related_components() -> None:
    assert olt2t.ChatTask == olt2t.models.ChatTask
    assert olt2t.ChatRoleType == olt2t.models.ChatRoleType
    assert olt2t.ChatTurn == olt2t.models.ChatTurn
