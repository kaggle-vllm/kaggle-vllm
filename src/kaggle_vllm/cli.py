from __future__ import annotations

import argparse

from .doctor import run_doctor, suggested_build_env
from .environment import as_json


def main() -> None:
    parser = argparse.ArgumentParser(prog="kaggle-vllm")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    sub.add_parser("fingerprint")
    sub.add_parser("build-env")
    args = parser.parse_args()

    if args.cmd == "doctor":
        raise SystemExit(run_doctor())
    if args.cmd == "fingerprint":
        print(as_json())
        return
    if args.cmd == "build-env":
        for key, value in suggested_build_env().items():
            print(f"export {key}={value}")
        return


if __name__ == "__main__":
    main()
