import logging

import beat
import beat.cli


def test_version(caplog):
    caplog.set_level(logging.INFO)
    ret = beat.cli.main(["version"])
    assert ret == 0
    assert caplog.records[0].msg == f"fenicsx-beat: {beat.__version__}"
