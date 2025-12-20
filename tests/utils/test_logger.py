"""
Copyright (c) 2025 MyoLab, Inc.

Released under the MyoLab Non-Commercial Scientific Research License
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

You may not use this file except in compliance with the License.
See the LICENSE file for governing permissions and limitations.
"""

# Summary: Tests for logger utils

from unittest.mock import MagicMock, patch

import pytest

from myo_tools.utils.log_ops.logger import Logger, getLogger


@pytest.fixture
def mock_logger():
    with patch("myo_tools.utils.log_ops.logger.logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock(spec=Logger)
        mock_logger.name = "test_logger"
        mock_get_logger.return_value = mock_logger
        yield mock_logger


def test_get_logger(mock_logger):
    logger = getLogger("test_logger")
    assert isinstance(logger, Logger)
    assert logger.name == "test_logger"
