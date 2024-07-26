import re

import olt2t


def test_olt2t_has_version() -> None:
    assert re.match(r"\d+\.\d+\.\d+", olt2t.__version__)
