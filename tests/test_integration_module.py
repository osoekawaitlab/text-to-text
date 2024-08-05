from collections.abc import Generator

import pytest

import olt2t


@pytest.mark.slow
def test_olt2t_as_module_chat() -> None:
    settings = olt2t.TextToTextCoreSettings(
        text_to_text_model_settings={"type": "PHI_3", "path_or_model_name": "microsoft/Phi-3-mini-4k-instruct"}
    )
    core = olt2t.TextToTextCore.create(settings=settings)
    task = olt2t.ChatTask(
        turns=[
            olt2t.ChatTurn(role=olt2t.ChatRoleType.USER, content=olt2t.StrT("How to make a cake?")),
            olt2t.ChatTurn(role=olt2t.ChatRoleType.ASSISTANT, content=olt2t.StrT("I don't know.")),
            olt2t.ChatTurn(role=olt2t.ChatRoleType.USER, content=olt2t.StrT("Okay. No problem. Bye.")),
        ]
    )
    actual = core(task)
    assert isinstance(actual, Generator)
    words = list(actual)
    assert len(words) >= 1
    assert all(isinstance(word, olt2t.StrT) for word in words)


@pytest.mark.slow
def test_olt2t_as_module_summarize() -> None:
    settings = olt2t.TextToTextCoreSettings(
        text_to_text_model_settings={"type": "PHI_3", "path_or_model_name": "microsoft/Phi-3-mini-4k-instruct"}
    )
    core = olt2t.TextToTextCore.create(settings=settings)
    task = olt2t.SummarizationTask(text=olt2t.StrT("How to make a cake?"), target_length=5)
    actual = core(task)
    assert isinstance(actual, Generator)
    words = list(actual)
    assert len(words) >= 1
    assert all(isinstance(word, olt2t.StrT) for word in words)
