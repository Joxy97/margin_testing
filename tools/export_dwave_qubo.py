#!/usr/bin/env python3
"""Convert a serialized D-Wave BQM/CQM (or LP CQM) to the C++ solver format."""

import argparse
import json
from pathlib import Path

import dimod
from dimod.serialization.fileview import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--penalty", type=float, default=None)
    args = parser.parse_args()

    source = Path(args.input)
    if source.suffix.lower() == ".lp":
        model = dimod.lp.load(str(source))
    else:
        with source.open("rb") as file:
            model = load(file)

    inverter = None
    if isinstance(model, dimod.ConstrainedQuadraticModel):
        model, inverter = dimod.cqm_to_bqm(model, lagrange_multiplier=args.penalty)
    if not isinstance(model, dimod.BinaryQuadraticModel):
        raise TypeError("input must contain a D-Wave BQM or CQM")

    bqm = model.binary
    labels = list(bqm.variables)
    indices = {label: i for i, label in enumerate(labels)}
    with Path(args.output).open("w", encoding="utf-8") as output:
        output.write(f"p qubo {len(labels)}\n")
        output.write(f"o {bqm.offset:.17g}\n")
        for label, bias in bqm.linear.items():
            if bias:
                output.write(f"l {indices[label]} {bias:.17g}\n")
        for (u, v), bias in bqm.quadratic.items():
            if bias:
                output.write(f"q {indices[u]} {indices[v]} {bias:.17g}\n")

    metadata = {"variables": labels}
    if inverter is not None:
        metadata["cqm_inverter"] = inverter.to_dict()
    with Path(args.output + ".metadata.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, default=repr)


if __name__ == "__main__":
    main()
