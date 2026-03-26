from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    try:
        from desktop_review.qt_app import run_app
    except ImportError as exc:  # pragma: no cover - depends on optional desktop deps
        message = (
            "PySide6 desktop dependencies are not installed. "
            "Install PySide6 to run the desktop review app."
        )
        print(message, file=sys.stderr)
        raise SystemExit(2) from exc
    run_app(argv)


if __name__ == "__main__":
    main()
