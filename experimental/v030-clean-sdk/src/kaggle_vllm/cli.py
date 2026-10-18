"""Launch SDK validation with CPython 3.12."""

import json
import os
import sys

PY312 = "/usr/bin/python3.12"

def main():
    if sys.version_info[:2] != (3, 12):
        if not os.path.isfile(PY312):
            raise SystemExit(
                f"Interpreter not found: {PY312}"
            )
        os.execv(
            PY312,
            [PY312, "-m", "kaggle_vllm.cli"],
        )

    from .runtime import check_runtime
    print(json.dumps(check_runtime(), indent=2))

if __name__ == "__main__":
    main()
