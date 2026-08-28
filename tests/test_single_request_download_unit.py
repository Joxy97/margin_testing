"""Tests for single-request download orchestration."""

import unittest
from datetime import date
from typing import Any

from download_unit import (
    Command,
    DataProvider,
    Period,
    SingleRequestDownloadUnit,
    UnifiedFormatCommand,
)


class StubDataProvider(DataProvider):
    def __init__(self) -> None:
        self.downloaded_command: Command | None = None

    def convertRawData(self, raw_data: Any) -> Any:
        return ("converted", raw_data)

    def convertCommand(self, command: UnifiedFormatCommand) -> Command:
        return command

    def downloadData(self, command: Command) -> Any:
        self.downloaded_command = command
        return "raw data"


class SingleRequestDownloadUnitTest(unittest.TestCase):
    def test_accepts_the_unified_command_dictionary(self) -> None:
        provider = StubDataProvider()
        command = UnifiedFormatCommand(
            instruments=[],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            period=Period.ONE_DAY,
        )
        unit = SingleRequestDownloadUnit()

        result = unit.getData(provider, command)

        self.assertIs(provider.downloaded_command, command)
        self.assertEqual(result, ("converted", "raw data"))
        self.assertFalse(hasattr(unit, "_provider"))


if __name__ == "__main__":
    unittest.main()
