"""Dense numeric storage for per-instrument return/volatility states."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy

from download_unit import Instrument


class DenseReturnsVolaGrid(Mapping[Instrument, numpy.ndarray]):
    """Store variable-length instrument grids in one padded numeric tensor.

    ``gridValues`` has shape ``(assets, padded_states, 2)`` and
    ``validStateMask`` identifies the real states. Mapping access is retained
    as a compatibility boundary, while performance-sensitive consumers can
    operate directly on the dense arrays.
    """

    def __init__(
        self,
        instruments: tuple[Instrument, ...],
        values: numpy.ndarray,
        validStateMask: numpy.ndarray,
        fallbackAssetMask: numpy.ndarray | None = None,
    ) -> None:
        instrument_order = tuple(instruments)
        if len(set(instrument_order)) != len(instrument_order):
            raise ValueError("grid instruments must be unique")
        dense_values = numpy.ascontiguousarray(values, dtype=float)
        valid_mask = numpy.ascontiguousarray(validStateMask, dtype=bool)
        fallback_mask = numpy.ascontiguousarray(
            numpy.zeros(len(instrument_order), dtype=bool)
            if fallbackAssetMask is None
            else fallbackAssetMask,
            dtype=bool,
        )
        if dense_values.ndim != 3 or dense_values.shape[2] != 2:
            raise ValueError(
                "dense return-volatility values must have shape "
                "(assets, states, 2)"
            )
        if valid_mask.shape != dense_values.shape[:2]:
            raise ValueError("validStateMask must match the asset/state shape")
        if dense_values.shape[0] != len(instrument_order):
            raise ValueError("dense values must contain one row per instrument")
        if fallback_mask.shape != (len(instrument_order),):
            raise ValueError("fallbackAssetMask must contain one value per asset")
        if not numpy.isfinite(dense_values[valid_mask]).all():
            raise ValueError("valid return-volatility states must be finite")

        state_counts = numpy.count_nonzero(valid_mask, axis=1).astype(
            numpy.int64,
            copy=False,
        )
        return_bounds = numpy.full((len(instrument_order), 2), numpy.nan)
        if dense_values.shape[1]:
            returns = dense_values[:, :, 0]
            nonempty = state_counts > 0
            return_bounds[nonempty, 0] = numpy.min(
                numpy.where(valid_mask[nonempty], returns[nonempty], numpy.inf),
                axis=1,
            )
            return_bounds[nonempty, 1] = numpy.max(
                numpy.where(valid_mask[nonempty], returns[nonempty], -numpy.inf),
                axis=1,
            )

        dense_values.setflags(write=False)
        valid_mask.setflags(write=False)
        state_counts.setflags(write=False)
        return_bounds.setflags(write=False)
        fallback_mask.setflags(write=False)
        self.instruments = instrument_order
        self.gridValues = dense_values
        self.validStateMask = valid_mask
        self.stateCounts = state_counts
        self.returnBounds = return_bounds
        self.fallbackAssetMask = fallback_mask
        self._instrumentIndices: dict[Instrument, int] | None = None

    @classmethod
    def fromMapping(
        cls,
        grids: Mapping[Instrument, numpy.ndarray] | DenseReturnsVolaGrid,
    ) -> DenseReturnsVolaGrid:
        """Pack a conventional instrument-to-grid mapping densely."""
        if isinstance(grids, cls):
            return grids
        instruments = tuple(grids)
        arrays = []
        for instrument in instruments:
            grid = numpy.asarray(grids[instrument], dtype=float)
            if grid.ndim != 2 or grid.shape[1] != 2:
                raise ValueError(
                    f"{instrument} risk states must have two columns"
                )
            arrays.append(grid)
        padded_states = max((len(grid) for grid in arrays), default=0)
        values = numpy.zeros((len(arrays), padded_states, 2), dtype=float)
        valid_mask = numpy.zeros((len(arrays), padded_states), dtype=bool)
        for asset, grid in enumerate(arrays):
            state_count = len(grid)
            values[asset, :state_count] = grid
            valid_mask[asset, :state_count] = True
        return cls(instruments, values, valid_mask)

    @property
    def validReturns(self) -> numpy.ndarray:
        """Return all valid returns in asset-major/state-major order."""
        return self.gridValues[:, :, 0][self.validStateMask]

    def __getitem__(self, instrument: Instrument) -> numpy.ndarray:
        if self._instrumentIndices is None:
            self._instrumentIndices = {
                item: index for index, item in enumerate(self.instruments)
            }
        asset = self._instrumentIndices[instrument]
        return self.gridValues[asset, self.validStateMask[asset]]

    def __iter__(self) -> Iterator[Instrument]:
        return iter(self.instruments)

    def __len__(self) -> int:
        return len(self.instruments)
