"""Tests for single-request download orchestration."""

import unittest
from datetime import date
from typing import Any

from download_unit import (
    Command,
    DataProvider,
    Period,
    SingleRequestDownloadUnit,
    DataRequest,
)


class StubDataProvider(DataProvider):
    def __init__(self) -> None:
        self.downloaded_command: Command | None = None

    def getDataTypes(self) -> set[str]:
        return {"testData"}

    def convertRawData(self, raw_data: Any) -> Any:
        return ("converted", raw_data)

    def convertCommand(self, command: DataRequest) -> Command:
        return command

    def downloadData(self, command: Command) -> Any:
        self.downloaded_command = command
        return "raw data"


class SingleRequestDownloadUnitTest(unittest.TestCase):
    def test_accepts_a_typed_data_request(self) -> None:
        provider = StubDataProvider()
        command = DataRequest(
            instruments=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            period=Period.ONE_DAY,
            data_type="testData",
        )
        unit = SingleRequestDownloadUnit()

        result = unit.getData(provider, command)

        self.assertIs(provider.downloaded_command, command)
        self.assertEqual(result, ("converted", "raw data"))
        self.assertFalse(hasattr(unit, "_provider"))


if __name__ == "__main__":
    unittest.main()
