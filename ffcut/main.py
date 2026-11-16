"""Entry point for ffcut.

``python -m ffcut.main`` / ``./ffcut-gui [file]``.
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication, QMessageBox

USAGE = """usage: ffcut-gui [VIDEO]

A PyQt6 cutter/renderer around ffmpeg: drop a video in, snip ranges out of the
timeline (cuts collapse), mute/balance the audio lanes, and render the whole
edit or just the selection — H.264 via VA-API/x264, or AV1 via VA-API/SVT-AV1.
"""


def main(argv: list[str] | None = None) -> int:
    """Build the app, optionally load a file from argv, run the event loop."""
    args = list(sys.argv if argv is None else argv)
    if any(arg in ("-h", "--help") for arg in args[1:]):
        print(USAGE)
        return 0

    app = QApplication(args)
    app.setApplicationName("ffcut")
    app.setApplicationDisplayName("ffcut")
    app.setOrganizationName("ffcut")
    app.setDesktopFileName("ffcut")

    try:
        from .app import MainWindow
    except ImportError as exc:  # engine half missing / broken import
        message = f"ffcut failed to start:\n\n{type(exc).__name__}: {exc}"
        print(message, file=sys.stderr)
        QMessageBox.critical(None, "ffcut", message)
        return 1

    window = MainWindow()
    window.show()

    for arg in args[1:]:
        if not arg.startswith("-"):
            window.load_file(arg)
            break

    code = app.exec()
    # Destroy the window (media player, video sink, worker threads) while the
    # QApplication is still alive.  Left to interpreter teardown, Python frees
    # the two in whatever order the collector picks, and Qt objects outliving
    # their application is how a clean exit turns into a segfault.
    window.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
