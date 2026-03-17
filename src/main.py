from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    # Support running this file directly (e.g. `py src/main.py` or `cd src; py main.py`).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.fastapi_gateway.runtime import run_gateway


def main() -> None:
    config = Config()
    run_gateway(config)


if __name__ == "__main__":
    main()
