"""Tests for single-request download orchestration."""

import unittest
from typing import Any

from download_unit import Command, DataProvider, SingleRequestDownloadUnit


class StubDataProvider(DataProvider):
    def __init__(self) -> None:
        self.downloaded_command: Command | None = None

    def convertRawData(self, raw_data: Any) -> Any:
        return ("converted", raw_data)

    def convertCommand(self, command: Command) -> Command:
        return command

    def downloadData(self, command: Command) -> Any:
        self.downloaded_command = command
        return "raw data"


class SingleRequestDownloadUnitTest(unittest.TestCase):
    def test_accepts_the_base_command_type(self) -> None:
        provider = StubDataProvider()
        command = Command()

        result = SingleRequestDownloadUnit().getData(provider, command)

        self.assertIs(provider.downloaded_command, command)
        self.assertEqual(result, ("converted", "raw data"))


if __name__ == "__main__":
    unittest.main()
