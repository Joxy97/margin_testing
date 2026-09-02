"""Tests for an independent download-provider registry."""

import unittest
from datetime import date
from unittest.mock import Mock, patch

from download_manager import DownloadManager
from download_unit import (
    DateChunker,
    InstrumentChunker,
    Period,
    ProductChunker,
    DataRequest,
    LocalCSVDataProvider,
    YfinanceDataProvider,
)


def data_request() -> DataRequest:
    return DataRequest(
        instruments=["AAPL"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        period=Period.ONE_DAY,
        data_type="closePrices",
    )


class DownloadManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = DownloadManager()
        self.manager.purgeProviders()

    def tearDown(self) -> None:
        self.manager.purgeProviders()

    def test_instances_keep_independent_provider_registries(self) -> None:
        provider = object()
        self.manager.addProvider("test", provider)  # type: ignore[arg-type]

        other_reference = DownloadManager()

        self.assertIsNot(other_reference, self.manager)
        self.assertNotIn("test", other_reference.providers)

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
        command = data_request()
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
                data_request(),
                "exponential_backoff",
                {"chunker": chunker, "time": 1},
            )

    def test_returns_capable_providers_and_prefers_local_csv(self) -> None:
        local = LocalCSVDataProvider()
        yfinance = YfinanceDataProvider()
        self.manager.addProviders(
            (("yfinance", yfinance), ("localCSV", local))
        )

        providers = self.manager.returnProviders("closePrices")
        selected = self.manager.providerSelection.selectProvider(providers)

        self.assertEqual(providers, [yfinance, local])
        self.assertIs(selected, local)

    def test_rejects_a_data_type_without_a_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "No provider"):
            self.manager.downloadDataType("missing", data_request())


if __name__ == "__main__":
    unittest.main()
