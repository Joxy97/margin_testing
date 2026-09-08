"""Observable retry behavior at the provider and clock seams."""

import unittest
from datetime import date
from unittest.mock import patch

import pandas

from download_unit import DataProvider, DataRequest, DateChunker, ExponentialBackoffDownloadUnit


class RemoteProvider(DataProvider):
    def __init__(self, responses):
        self.responses = iter(responses)

    def getDataTypes(self):
        return {"closePrices"}

    def convertCommand(self, command):
        return command

    def convertRawData(self, data):
        return data

    def downloadData(self, command):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return pandas.DataFrame({"date": [command.start_date], "A": [response]})


class DownloadRetryTest(unittest.TestCase):
    def test_optional_success_pacing_is_constant_without_a_trailing_sleep(self):
        request = self.request.withChanges(end_date=date(2024, 1, 3))
        with patch("download_unit.exponential_backoff_download_unit.sleep") as sleep:
            result = ExponentialBackoffDownloadUnit(DateChunker(1), time=1, requestInterval=2).getData(
                RemoteProvider([10, 11, 12]), request)
        self.assertEqual(result["A"].tolist(), [10, 11, 12])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 2])

    def test_yahoo_confirms_an_empty_interval_from_a_valid_wider_response(self):
        from download_unit.data_provider.yfinance import YfinanceDataProvider
        from yfinance.exceptions import YFPricesMissingError
        request = self.request.withChanges(start_date=date(2024, 1, 6), end_date=date(2024, 1, 6))
        prior = pandas.DataFrame({"Close": [10.]}, index=pandas.to_datetime(["2024-01-05"]))
        with patch("download_unit.data_provider.yfinance.yfinance.Ticker") as ticker:
            ticker.return_value.history.side_effect = [YFPricesMissingError("A", "empty"), prior]
            result = ExponentialBackoffDownloadUnit(DateChunker(1), time=0).getData(
                YfinanceDataProvider(), request)
            self.assertTrue(result.empty)
            self.assertEqual(result.columns.tolist(), ["date", "A"])
            ticker.return_value.history.side_effect = [YFPricesMissingError("A", "malformed")] * 2
            with self.assertRaises(YFPricesMissingError):
                ExponentialBackoffDownloadUnit(DateChunker(1), time=0).getData(
                    YfinanceDataProvider(), request)

    def test_yahoo_rate_limits_are_classified_by_the_provider(self):
        from download_unit.data_provider.yfinance import YfinanceDataProvider
        from yfinance.exceptions import YFRateLimitError
        raw = pandas.DataFrame({"Close": [10.]}, index=pandas.to_datetime(["2024-01-01"]))
        with patch("download_unit.data_provider.yfinance.yfinance.download", side_effect=AssertionError("bulk download hides failures")), \
             patch("download_unit.data_provider.yfinance.yfinance.Ticker") as ticker, \
             patch("download_unit.exponential_backoff_download_unit.sleep"):
            def history(**kwargs):
                if not kwargs.get("raise_errors"):
                    return raw.iloc[:0]
                if history.calls == 0:
                    history.calls += 1
                    raise YFRateLimitError()
                return raw
            history.calls = 0
            ticker.return_value.history.side_effect = history
            result = ExponentialBackoffDownloadUnit(DateChunker(1), time=0).getData(
                YfinanceDataProvider(), self.request)
        self.assertEqual(result["A"].tolist(), [10.])

    request = DataRequest(instruments=["A"], start_date=date(2024, 1, 1),
                          end_date=date(2024, 1, 1), data_type="closePrices")

    def test_transient_failure_recovers_without_trailing_sleep(self):
        with patch("download_unit.exponential_backoff_download_unit.sleep") as sleep:
            result = ExponentialBackoffDownloadUnit(DateChunker(1), time=1).getData(
                RemoteProvider([TimeoutError("temporarily unavailable"), 10]), self.request)
        self.assertEqual(result["A"].tolist(), [10])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1])

    def test_retry_exhaustion_caps_delays_and_propagates_last_error(self):
        error = TimeoutError("last attempt")
        with patch("download_unit.exponential_backoff_download_unit.sleep") as sleep:
            with self.assertRaises(TimeoutError) as caught:
                ExponentialBackoffDownloadUnit(
                    DateChunker(1), time=2, maxAttempts=4, maxDelay=3,
                ).getData(RemoteProvider([TimeoutError(), TimeoutError(), TimeoutError(), error]),
                          self.request)
        self.assertIs(caught.exception, error)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 3, 3])

    def test_server_hint_and_jitter_remain_capped(self):
        class RateLimitedProvider(RemoteProvider):
            def retryAfter(self, error):
                return 9.0

        with patch("download_unit.exponential_backoff_download_unit.sleep") as sleep, \
             patch("download_unit.exponential_backoff_download_unit.uniform", return_value=0.5):
            result = ExponentialBackoffDownloadUnit(
                DateChunker(1), time=2, maxDelay=5, jitter=0.5,
            ).getData(RateLimitedProvider([TimeoutError(), 10]), self.request)
        self.assertEqual(result["A"].tolist(), [10])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5])

    def test_invalid_retry_limits_fail_before_downloading(self):
        for parameters in ({"maxAttempts": 0}, {"maxDelay": -1},
                           {"jitter": float("nan")}, {"time": float("inf")}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                ExponentialBackoffDownloadUnit(DateChunker(1), **{"time": 1, **parameters})

    def test_permanent_failure_and_cancellation_do_not_sleep(self):
        for error in (ValueError("invalid response"), KeyboardInterrupt()):
            with self.subTest(error=error), \
                 patch("download_unit.exponential_backoff_download_unit.sleep") as sleep:
                class FailedProvider(RemoteProvider):
                    def downloadData(self, command):
                        raise error
                with self.assertRaises(type(error)):
                    ExponentialBackoffDownloadUnit(DateChunker(1), time=1).getData(
                        FailedProvider([]), self.request)
                sleep.assert_not_called()
