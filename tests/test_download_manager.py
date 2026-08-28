"""Tests for the singleton download-provider registry."""

import unittest
from datetime import date
from unittest.mock import Mock, patch

from download_manager import DownloadManager
from download_unit import (
    DateChunker,
    InstrumentChunker,
    Period,
    ProductChunker,
    UnifiedFormatCommand,
)


def unified_command() -> UnifiedFormatCommand:
    return UnifiedFormatCommand(
        instruments=[],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        period=Period.ONE_DAY,
    )


class DownloadManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = DownloadManager()
        self.manager.purgeProviders()

    def tearDown(self) -> None:
        self.manager.purgeProviders()

    def test_is_a_singleton_and_keeps_registered_providers(self) -> None:
        provider = object()
        self.manager.addProvider("test", provider)  # type: ignore[arg-type]

        other_reference = DownloadManager()

        self.assertIs(other_reference, self.manager)
        self.assertIs(other_reference.providers["test"], provider)

    def test_adds_one_or_multiple_providers(self) -> None:
        first = object()
        second = object()

        self.manager.addProvider("first", first)  # type: ignore[arg-type]
        self.manager.addProviders(  # type: ignore[arg-type]
            (("second", second), ("replacement", first))
        )
        self.manager.addProvider("replacement", second)  # type: ignore[arg-type]

        self.assertEqual(
            self.manager.providers,
            {"first": first, "second": second, "replacement": second},
        )

    def test_removes_one_or_multiple_providers(self) -> None:
        self.manager.addProviders(  # type: ignore[arg-type]
            (("first", object()), ("second", object()), ("third", object()))
        )

        self.manager.removeProvider("missing")
        self.manager.removeProvider("first")
        self.manager.removeProviders(["second", "also-missing"])

        self.assertEqual(set(self.manager.providers), {"third"})

    def test_purges_all_providers(self) -> None:
        self.manager.addProviders(  # type: ignore[arg-type]
            (("first", object()), ("second", object()))
        )

        self.manager.purgeProviders()

        self.assertEqual(self.manager.providers, {})

    @patch(
        "download_manager.download_manager.DownloadUnitFactory.createDownloadUnit"
    )
    def test_downloads_with_selected_provider_and_algorithm(
        self,
        create_download_unit: Mock,
    ) -> None:
        provider = object()
        command = unified_command()
        chunker = ProductChunker(InstrumentChunker(5), DateChunker(100))
        download_unit = create_download_unit.return_value
        download_unit.getData.return_value = "downloaded data"
        self.manager.addProvider("provider", provider)  # type: ignore[arg-type]

        result = self.manager.downloadData(
            "provider",
            command,
            "exponential_backoff",
            {"chunker": chunker, "time": 1},
        )

        create_download_unit.assert_called_once_with(
            "exponential_backoff",
            {"chunker": chunker, "time": 1},
        )
        download_unit.getData.assert_called_once_with(provider, command)
        self.assertEqual(result, "downloaded data")

    def test_download_rejects_missing_provider(self) -> None:
        chunker = ProductChunker(InstrumentChunker(5), DateChunker(100))
        with self.assertRaises(KeyError):
            self.manager.downloadData(
                "missing",
                unified_command(),
                "exponential_backoff",
                {"chunker": chunker, "time": 1},
            )


if __name__ == "__main__":
    unittest.main()
