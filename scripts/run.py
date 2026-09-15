"""Entry point — run the API server.

Usage:
    python scripts/run.py            # 0.0.0.0:8000
    python scripts/run.py --port 9000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run langchain-ayaka API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Ensure the project root is importable when run as a plain script.
    root = Path(__file__).resolve().parents[1]
    import sys

    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))

    uvicorn.run(
        "ayaka.api.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
