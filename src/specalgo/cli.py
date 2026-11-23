"""Run an algorithm on local inputs and write its JSON result."""

import argparse
import json
import os
from pathlib import Path

from .common import read_json, writable, work_root, write_json


def main():
    parser = argparse.ArgumentParser(
        prog="specalgo", description="Run specification algorithms on local resources."
    )
    parser.add_argument("algorithm", choices=["alignment-value", "path-influence"])
    parser.add_argument("--request", required=True, type=Path, help="JSON request")
    parser.add_argument(
        "--output", required=True, type=Path, help="destination for the JSON result"
    )
    parser.add_argument(
        "--resources",
        type=Path,
        help="read-only resource directory (default: SPECALGO_RESOURCES or ./resources)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="run state and caches (default: SPECALGO_WORK_DIR or ./runs)",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        help="external scored training observations in JSON Lines format for represent",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help="external JSON object with selected_case_ids for represent",
    )
    args = parser.parse_args()
    if args.resources is not None:
        os.environ["SPECALGO_RESOURCES"] = str(args.resources)
    if args.work_dir is not None:
        os.environ["SPECALGO_WORK_DIR"] = str(args.work_dir)
    from .algorithms import PathInfluenceAlgorithm, SpecAlignmentValueAlgorithm
    from .contracts import Algorithm1Request, Algorithm2Request

    try:
        work_root()
        output = writable(args.output)
        inputs = [args.request, args.observations, args.selection]
        if any(path is not None and path.resolve() == output for path in inputs):
            raise ValueError(
                "Choose an output distinct from the request, observations and selection"
            )
        request = read_json(args.request)
        if args.algorithm == "alignment-value":
            validated = Algorithm1Request.model_validate(request)
            if validated.operation != "represent" and (
                args.observations or args.selection
            ):
                raise ValueError(
                    "Observation and selection files apply only to represent"
                )
            algorithm = SpecAlignmentValueAlgorithm(
                observations=args.observations, selection=args.selection
            )
        else:
            validated = Algorithm2Request.model_validate(request)
            if args.observations or args.selection:
                raise ValueError(
                    "Observation and selection files apply only to represent"
                )
            algorithm = PathInfluenceAlgorithm()
        result = algorithm.run(validated)
        write_json(output, result)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    print(json.dumps({"status": "completed", "output": str(args.output)}))


if __name__ == "__main__":
    main()
