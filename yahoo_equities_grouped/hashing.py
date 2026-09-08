"""Deterministic polynomial hashing for trading-day sequences."""

from collections.abc import Iterable, Mapping

PRIME = 1_000_000_007
BASE = 911_382_323


def polynomial_hash(values: Iterable[int], *, prime: int = PRIME, base: int = BASE) -> str:
    """Return the polynomial hash of an ordered sequence as a string key."""
    result = 0
    for value in values:
        result = (result * base + value) % prime
    return str(result)


def date_numbers(dates: Iterable[str], date_to_number: Mapping[str, int]) -> list[int]:
    """Translate dates to their global, one-based numeric representation."""
    try:
        return [date_to_number[date] for date in dates]
    except KeyError as error:
        raise ValueError(f"Date {error.args[0]!r} is missing from the global date map") from error
