from olt2t import StrT
from olt2t.text_to_text_models.echo import EchoTextToTextModel


def test_echo_text_to_text_model_generate() -> None:
    model = EchoTextToTextModel()
    actual = model.generate(StrT("How to make a cake?"))
    assert list(actual) == [StrT("How to make a cake?")]
