import logging

import beat
from beat.cli import main


def test_version(caplog):
    caplog.set_level(logging.INFO)
    ret = main(["version"])
    assert ret == 0
    assert caplog.records[0].msg == f"fenicsx-beat: {beat.__version__}"
