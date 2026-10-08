"""Read one value from a checked release protocol."""

from __future__ import annotations

import argparse
import json

from tuco.protocol import value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setting", choices=("single_sim", "sim2sim", "sim2real"))
    parser.add_argument("task")
    parser.add_argument("key")
    args = parser.parse_args()
    result = value(args.setting, args.task, args.key)
    if isinstance(result, (dict, list)):
        print(json.dumps(result, separators=(",", ":")))
    elif isinstance(result, bool):
        print("true" if result else "false")
    else:
        print(result)


if __name__ == "__main__":
    main()

