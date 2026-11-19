"""Root shim so ``python -m dlfs`` works from the repository root."""

import os
import runpy
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

if __name__ == "__main__":
    runpy.run_module("dlfs", run_name="__main__", alter_sys=True)