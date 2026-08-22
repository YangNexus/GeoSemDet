import inspect
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.logger_module import get_logger


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_distributed_logger_reports_business_callsite():
    logger = get_logger("tests.engine.logger_module.stacklevel")
    handler = _CaptureHandler()
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.is_dist_initialized = True

    expected_lineno = inspect.currentframe().f_lineno + 1
    logger.info("stacklevel check")

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.filename == Path(__file__).name
    assert record.funcName == "test_distributed_logger_reports_business_callsite"
    assert record.lineno == expected_lineno
